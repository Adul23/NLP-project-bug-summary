"""CodeBERT/RoBERTa LoRA classification on the frozen shared splits."""

import argparse
import time
from pathlib import Path

from .experiment_data import DEFAULT_SPLITS, load_splits
from .lora_common import (
    LORA_TARGETS,
    PRECISIONS,
    add_classifier_lora,
    class_weights,
    hardware_options,
    loader_options,
    pad_multiple,
    require_empty,
    runtime_report,
    softmax_bug_probability,
    tokenize_once,
    tune_threshold,
    weighted_trainer,
    write_json,
)
from .split_dataset import ROOT
from .train_tfidf import evaluate_predictions

DEFAULTS = {
    "eval_batch_size": 64,
    "precision": "auto",
    "class_weights": False,
    "lora_targets": "qv",
    "lora_r": 8,
    "threshold_tune": False,
}


def train(args):
    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    # Callers that build a Namespace by hand (tests, notebooks) may omit newer options.
    for name, value in DEFAULTS.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    started = time.perf_counter()
    options = hardware_options(args.device, args.precision)
    cuda = not options["use_cpu"]
    if cuda:
        torch.cuda.reset_peak_memory_stats()
        if args.batch_size == 1:
            print(
                "Hint: --batch-size 1 on CUDA is slow; try --batch-size 16 --accumulation 1 "
                "(or 8 with --accumulation 2 if memory is tight)."
            )
    set_seed(42)
    (training, validation, test), manifest = load_splits(args.split_dir)
    output = require_empty(args.output)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=args.revision)
    base = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        revision=args.revision,
        num_labels=2,
        id2label={0: "CLEAN", 1: "BUGGY"},
        label2id={"CLEAN": 0, "BUGGY": 1},
    )
    model, targets = add_classifier_lora(base, args.lora_targets, args.lora_r)
    model.print_trainable_parameters()
    encoded = []
    truncation = {}
    for name, frame in zip(
        ("train", "validation", "test"), (training, validation, test)
    ):
        # One tokenizer pass per split: lengths, truncation statistics and truncated ids.
        features, truncation[name] = tokenize_once(
            tokenizer, frame.code.tolist(), args.max_length
        )
        encoded.append(Dataset.from_dict({**features, "labels": frame.label.tolist()}))
    config = TrainingArguments(
        output_dir=str(output / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.accumulation,
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        warmup_ratio=0.06,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="bug_f1",
        greater_is_better=True,
        save_total_limit=1,
        seed=42,
        data_seed=42,
        report_to="none",
        group_by_length=True,  # batches of similar length waste far less padding
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        **loader_options(options),
        **options,
    )

    # Keep every validation pass (full report + probabilities) keyed by training step, so the
    # best checkpoint's validation result is reused instead of predicting validation again.
    latest, by_step = {}, {}

    def compute_metrics(result):
        report = evaluate_predictions(
            result.label_ids, np.argmax(result.predictions, axis=-1)
        )
        latest["report"] = report
        latest["labels"] = np.asarray(result.label_ids)
        latest["bug_probability"] = softmax_bug_probability(result.predictions)
        return {
            k: v for k, v in report.items() if k.startswith("bug_") or k == "accuracy"
        }

    class KeepValidation(TrainerCallback):
        def on_evaluate(self, args, state, control, **kwargs):
            if latest:
                by_step[state.global_step] = dict(latest)

    weights = class_weights(training.label.tolist()) if args.class_weights else None
    trainer_class = weighted_trainer(weights) if weights else Trainer
    trainer = trainer_class(
        model=model,
        args=config,
        train_dataset=encoded[0],
        eval_dataset=encoded[1],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=pad_multiple(options)
        ),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2), KeepValidation()],
    )
    try:
        outcome = trainer.train()
    except torch.OutOfMemoryError as exc:
        raise RuntimeError(
            "Out of memory: lower --batch-size (e.g. 8 with --accumulation 2, or 1 with "
            "--accumulation 16), add --gradient-checkpointing, or use --device cpu"
        ) from exc
    best = trainer.state.best_model_checkpoint
    best_step = int(str(best).rsplit("-", 1)[-1]) if best else None
    kept = by_step.get(best_step)
    if kept is None:  # no recorded pass for the loaded checkpoint: evaluate once
        trainer.evaluate()
        kept = dict(latest)
    final = trainer.predict(
        encoded[2]
    )  # Once, after validation-selected checkpoint loading.
    test_argmax = evaluate_predictions(final.label_ids, final.predictions.argmax(-1))
    validation_report, test_report, threshold = kept["report"], test_argmax, None
    if args.threshold_tune:
        threshold, _ = tune_threshold(kept["labels"], kept["bug_probability"])
        validation_report = evaluate_predictions(
            kept["labels"], (kept["bug_probability"] >= threshold).astype(int)
        )
        test_report = evaluate_predictions(
            final.label_ids,
            (softmax_bug_probability(final.predictions) >= threshold).astype(int),
        )
    model.save_pretrained(output / "adapter")
    tokenizer.save_pretrained(output / "adapter")
    metadata = {
        "base_model": args.base_model,
        "revision": getattr(base.config, "_commit_hash", None) or args.revision,
        "max_length": args.max_length,
        "targets": targets,
        "head": "classifier",
        "split_manifest": manifest,
        "seed": 42,
        "epochs": args.epochs,
        "device": str(trainer.args.device),
        "truncation_fraction": truncation,
        "best_checkpoint": best,
        "status": "trained",
        "hyperparameters": {
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "accumulation": args.accumulation,
            "learning_rate": 2e-4,
            "lr_scheduler": "linear",
            "warmup_ratio": 0.06,
            "group_by_length": True,
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "lora_targets": args.lora_targets,
            "lora_r": args.lora_r,
            "class_weights": weights,
        },
        "runtime": runtime_report(outcome.metrics, started, options, len(encoded[0])),
    }
    if threshold is not None:
        metadata["decision_threshold"] = threshold
    write_json(output / "training_config.json", metadata)
    report = {
        "model": "CodeBERT + LoRA",
        "status": "trained",
        "split_manifest": manifest,
        "validation": validation_report,
        "test": test_report,
    }
    if threshold is not None:
        report["decision_threshold"] = threshold
        report["validation_argmax"] = kept["report"]
        report["test_argmax"] = test_argmax
    write_json(output / "metrics.json", report)
    print(report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-dir", type=Path, default=DEFAULT_SPLITS)
    p.add_argument("--output", type=Path, default=ROOT / "models/codebert_lora")
    p.add_argument("--base-model", default="microsoft/codebert-base")
    p.add_argument("--revision", default="main")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument(
        "--precision",
        choices=PRECISIONS,
        default="auto",
        help="auto: bf16 if the GPU supports it natively, else fp16 on CUDA, fp32 on CPU",
    )
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--accumulation", type=int, default=1)
    p.add_argument("--max-length", type=int, default=512, metavar="8..512")
    p.add_argument("--gradient-checkpointing", action="store_true")
    quality = p.add_argument_group(
        "quality options (off by default so runs stay comparable)"
    )
    quality.add_argument(
        "--class-weights",
        action="store_true",
        help="class-weighted cross-entropy for the clean/buggy imbalance",
    )
    quality.add_argument(
        "--lora-targets", choices=[*LORA_TARGETS, "all-linear"], default="qv"
    )
    quality.add_argument("--lora-r", type=int, default=8)
    quality.add_argument(
        "--threshold-tune",
        action="store_true",
        help="choose the BUGGY decision threshold on validation and store it",
    )
    a = p.parse_args()
    if min(a.epochs, a.batch_size, a.eval_batch_size, a.accumulation, a.lora_r) < 1:
        p.error("epochs, batch sizes, accumulation and LoRA rank must be positive")
    if not 8 <= a.max_length <= 512:
        p.error("max length must be between 8 and 512")
    train(a)


if __name__ == "__main__":
    main()
