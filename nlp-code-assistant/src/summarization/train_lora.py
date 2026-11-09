"""CodeT5 LoRA on verified synthetic pair explanations, not human bug summaries."""

import argparse
import json
from pathlib import Path

from src.bug_detection.experiment_data import sha256
from src.bug_detection.lora_common import (
    add_summarizer_lora,
    hardware_options,
    require_empty,
    write_json,
)
from src.bug_detection.split_dataset import ROOT
from src.summarization.prepare import verified_change

PROMPT = "Describe the potential synthetic operator mutation in this Python function:\n"


def load_supervision(folder):
    manifest = json.loads((Path(folder) / "manifest.json").read_text())
    parts = []
    seen = set()
    for name in ("train", "validation", "test"):
        if sha256(Path(folder) / f"{name}.jsonl") != manifest["files"][name]:
            raise ValueError("Supervision explanation file differs from its manifest")
        rows = [
            json.loads(line)
            for line in (Path(folder) / f"{name}.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if not rows:
            raise ValueError(f"Empty {name} supervision")
        ids = {row["source_id"] for row in rows}
        if seen & ids:
            raise ValueError("Supervision source_id leakage")
        seen |= ids
        for row in rows:
            change = verified_change(row["original_code"], row["buggy_code"])
            if (
                row.get("supervision") != "verified_synthetic_pair"
                or row["explanation"] != change["explanation"]
            ):
                raise ValueError("Unverified or modified synthetic explanation")
        parts.append(rows)
    return parts


def train(args):
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    )

    options = hardware_options(args.device)
    set_seed(42)
    parts = load_supervision(args.data_dir)
    output = require_empty(args.output)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=args.revision)
    base = AutoModelForSeq2SeqLM.from_pretrained(
        args.base_model, revision=args.revision
    )
    model, targets = add_summarizer_lora(base)
    base.config.use_cache = False
    encoded = []
    for rows in parts:
        data = Dataset.from_list(rows)

        def tokenize(batch):
            inputs = tokenizer(
                [PROMPT + code for code in batch["buggy_code"]],
                truncation=True,
                max_length=args.max_length,
                return_token_type_ids=False,
            )
            inputs["labels"] = tokenizer(
                text_target=batch["explanation"], truncation=True, max_length=128
            )["input_ids"]
            return inputs

        encoded.append(
            data.map(tokenize, batched=True, remove_columns=data.column_names)
        )
    config = Seq2SeqTrainingArguments(
        output_dir=str(output / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation,
        learning_rate=2e-4,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        seed=42,
        data_seed=42,
        report_to="none",
        dataloader_num_workers=0,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        **options,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=config,
        train_dataset=encoded[0],
        eval_dataset=encoded[1],
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    try:
        trainer.train()
    except torch.OutOfMemoryError as exc:
        raise RuntimeError(
            "Reduce --batch-size, enable --gradient-checkpointing, or use --device cpu"
        ) from exc
    trainer.remove_callback(EarlyStoppingCallback)
    test_metrics = trainer.evaluate(encoded[2], metric_key_prefix="test")
    model.save_pretrained(output / "adapter")
    tokenizer.save_pretrained(output / "adapter")
    write_json(
        output / "training_config.json",
        {
            "base_model": args.base_model,
            "revision": getattr(base.config, "_commit_hash", None) or args.revision,
            "max_length": args.max_length,
            "targets": targets,
            "seed": 42,
            "epochs": args.epochs,
            "status": "trained",
            "supervision": "verified_synthetic_pair",
            "human_annotated": False,
            "source_manifest": json.loads(
                (args.data_dir / "manifest.json").read_text()
            ),
            "best_checkpoint": trainer.state.best_model_checkpoint,
        },
    )
    write_json(
        output / "metrics.json",
        {
            "model": "CodeT5 + LoRA",
            "test": test_metrics,
            "interpretation": "Token loss on synthetic templates, not evidence of semantic bug explanation quality",
        },
    )
    # A fixed review sample, with references and blank human ratings, avoids inventing quality scores.
    model.eval()
    review = []
    for row in parts[2][:20]:
        inputs = tokenizer(
            PROMPT + row["buggy_code"],
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
            return_token_type_ids=False,
        ).to(model.device)
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        review.append(
            {
                "source_id": row["source_id"],
                "buggy_code": row["buggy_code"],
                "reference": row["explanation"],
                "generated": tokenizer.decode(ids[0], skip_special_tokens=True),
                "correctness_0_to_2": None,
                "specificity_0_to_2": None,
                "unsupported_claims": None,
            }
        )
    write_json(output / "human_review.json", review)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir", type=Path, default=ROOT / "data/processed/summaries_v1"
    )
    p.add_argument("--output", type=Path, default=ROOT / "models/codet5_lora")
    p.add_argument("--base-model", default="Salesforce/codet5-small")
    p.add_argument("--revision", default="main")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accumulation", type=int, default=8)
    p.add_argument("--max-length", type=int, default=512, metavar="8..512")
    p.add_argument("--gradient-checkpointing", action="store_true")
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.accumulation) < 1:
        p.error("epochs, batch size and accumulation must be positive")
    if not 8 <= args.max_length <= 512:
        p.error("max length must be between 8 and 512")
    train(args)


if __name__ == "__main__":
    main()
