"""Full fine-tuning of CodeT5-small for one-sentence code summaries (MPS/CUDA/CPU).

Uses the shared split stored in summarization_dataset.csv.
Only the train and validation splits are read; the test split is never touched here.
"""

import argparse
import json
import random
import tempfile
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL = "Salesforce/codet5-small"


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_split(path, name, limit=None):
    frame = pd.read_csv(path, dtype={"source_id": str}, keep_default_na=False)
    part = frame[frame["split"] == name]
    if limit:
        part = part.head(limit)
    if part.empty:
        raise ValueError(f"No rows for split '{name}'")
    return part["code"].tolist(), part["summary"].tolist()


def make_batches(codes, summaries, tokenizer, batch_size, max_len, max_target, shuffle, rng):
    order = list(range(len(codes)))
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        inputs = tokenizer(
            [codes[i] for i in idx],
            max_length=max_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        labels = tokenizer(
            [summaries[i] for i in idx],
            max_length=max_target,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).input_ids
        labels[labels == tokenizer.pad_token_id] = -100
        yield inputs, labels


def evaluate_loss(model, data, tokenizer, args, device, rng):
    model.eval()
    total, count = 0.0, 0
    with torch.inference_mode():
        for inputs, labels in make_batches(
            *data, tokenizer, args.batch_size, args.max_length, args.max_target, False, rng
        ):
            inputs = inputs.to(device)
            loss = model(**inputs, labels=labels.to(device)).loss
            total += loss.item() * labels.shape[0]
            count += labels.shape[0]
    return total / count


def train(args):
    random.seed(42)
    torch.manual_seed(42)
    rng = random.Random(42)
    device = pick_device()

    if args.smoke:
        output = Path(tempfile.mkdtemp(prefix="codet5_smoke_"))
        train_limit, val_limit, epochs = 64, 32, 1
    else:
        output = Path(args.output)
        train_limit, val_limit, epochs = None, None, args.epochs
        if output.exists() and any(output.iterdir()):
            raise ValueError(f"{output} is not empty; choose another --output")
    output.mkdir(parents=True, exist_ok=True)

    train_data = load_split(args.dataset, "train", train_limit)
    val_data = load_split(args.dataset, "validation", val_limit)
    print(f"device: {device} | train: {len(train_data[0])} | validation: {len(val_data[0])}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(device)

    steps_per_epoch = -(-len(train_data[0]) // args.batch_size)
    total_steps = steps_per_epoch * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(0.05 * total_steps), total_steps
    )

    history, best_loss, best_epoch, bad_epochs = [], float("inf"), 0, 0
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start, running, seen = time.time(), 0.0, 0
        for step, (inputs, labels) in enumerate(
            make_batches(
                *train_data, tokenizer, args.batch_size, args.max_length,
                args.max_target, True, rng,
            ),
            start=1,
        ):
            inputs = inputs.to(device)
            loss = model(**inputs, labels=labels.to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item() * labels.shape[0]
            seen += labels.shape[0]
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{steps_per_epoch} | train loss {running / seen:.3f}")

        train_loss = running / seen
        val_loss = evaluate_loss(model, val_data, tokenizer, args, device, rng)
        minutes = (time.time() - epoch_start) / 60
        print(f"epoch {epoch}: train {train_loss:.3f} | val {val_loss:.3f} | {minutes:.1f} min")
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "minutes": minutes}
        )

        if val_loss < best_loss:
            best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            model.save_pretrained(output / "best")
            tokenizer.save_pretrained(output / "best")
            print("  saved best model")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stopping (no improvement for {args.patience} epochs)")
                break

    summary = {
        "base_model": BASE_MODEL,
        "seed": 42,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "max_target": args.max_target,
        "learning_rate": args.lr,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "total_minutes": (time.time() - started) / 60,
        "history": history,
        "test_split_used": False,
    }
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"done. best epoch {best_epoch}, val loss {best_loss:.3f}, saved in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "data/processed/summarization_dataset.csv"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "models/codet5_summarizer")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-target", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--smoke", action="store_true", help="tiny run in a temp folder")
    train(parser.parse_args())


if __name__ == "__main__":
    main()