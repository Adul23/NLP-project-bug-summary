"""CodeBERT/RoBERTa LoRA classification on the frozen shared splits."""

import argparse
from pathlib import Path

from .experiment_data import DEFAULT_SPLITS, load_splits
from .lora_common import (
    add_classifier_lora,
    hardware_options,
    require_empty,
    write_json,
)
from .split_dataset import ROOT
from .train_tfidf import evaluate_predictions


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
        TrainingArguments,
        set_seed,
    )

    options = hardware_options(args.device)
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
    model, targets = add_classifier_lora(base)
    model.print_trainable_parameters()
    encoded = []
    truncation = {}
    for name, frame in zip(
        ("train", "validation", "test"), (training, validation, test)
    ):
        lengths = tokenizer(frame.code.tolist(), truncation=False, verbose=False)[
            "input_ids"
        ]
        truncation[name] = sum(len(ids) > args.max_length for ids in lengths) / len(
            lengths
        )
        data = Dataset.from_dict(
            {"code": frame.code.tolist(), "labels": frame.label.tolist()}
        )
        encoded.append(
            data.map(
                lambda batch: tokenizer(
                    batch["code"], truncation=True, max_length=args.max_length
                ),
                batched=True,
                remove_columns=["code"],
            )
        )
    config = TrainingArguments(
        output_dir=str(output / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation,
        learning_rate=2e-4,
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
        dataloader_num_workers=0,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        **options,
    )
    trainer = Trainer(
        model=model,
        args=config,
        train_dataset=encoded[0],
        eval_dataset=encoded[1],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=lambda result: {
            k: v
            for k, v in evaluate_predictions(
                result.label_ids, np.argmax(result.predictions, axis=-1)
            ).items()
            if k.startswith("bug_") or k == "accuracy"
        },
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    try:
        trainer.train()
    except torch.OutOfMemoryError as exc:
        raise RuntimeError(
            "Out of memory: use --batch-size 1 --accumulation 8 --gradient-checkpointing or --device cpu"
        ) from exc
    val = trainer.predict(encoded[1])
    final = trainer.predict(
        encoded[2]
    )  # Once, after validation-selected checkpoint loading.
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
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "status": "trained",
    }
    write_json(output / "training_config.json", metadata)
    report = {
        "model": "CodeBERT + LoRA",
        "status": "trained",
        "split_manifest": manifest,
        "validation": evaluate_predictions(val.label_ids, val.predictions.argmax(-1)),
        "test": evaluate_predictions(final.label_ids, final.predictions.argmax(-1)),
    }
    write_json(output / "metrics.json", report)
    print(report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-dir", type=Path, default=DEFAULT_SPLITS)
    p.add_argument("--output", type=Path, default=ROOT / "models/codebert_lora")
    p.add_argument("--base-model", default="microsoft/codebert-base")
    p.add_argument("--revision", default="main")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accumulation", type=int, default=8)
    p.add_argument("--max-length", type=int, default=512, metavar="8..512")
    p.add_argument("--gradient-checkpointing", action="store_true")
    a = p.parse_args()
    if min(a.epochs, a.batch_size, a.accumulation) < 1:
        p.error("epochs, batch size and accumulation must be positive")
    if not 8 <= a.max_length <= 512:
        p.error("max length must be between 8 and 512")
    train(a)


if __name__ == "__main__":
    main()
