"""Full fine-tuning of CodeT5-small for one-sentence code summaries (MPS/CUDA/CPU).

Uses the shared split stored in summarization_dataset.csv.
Only the train and validation splits are read; the test split is never touched here.

Outputs (default: checkpoints/codet5_summarizer, outside git):
  best/                  best model by validation loss
  train_log.jsonl        one JSON line per logged step and per epoch
  training_summary.json  settings, history, timing (rewritten after every epoch)

Quick smoke test with timing:
  python -m src.summarization.train_codet5 --limit 64 --epochs 1 --log-every 1
"""

import argparse
import json
import random
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
DEFAULT_OUTPUT = ROOT / "checkpoints/codet5_summarizer"


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sync(device):
    """Wait for queued GPU work so wall-clock step times are real."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def append_log(path, record):
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def resolve_output(args):
    if args.output is not None:
        return args.output
    if args.limit:
        # Smoke runs never reuse (or pollute) the real checkpoint folder.
        stamp = time.strftime("%Y%m%d_%H%M%S")
        return ROOT / f"checkpoints/codet5_smoke_{stamp}"
    return DEFAULT_OUTPUT


def train(args):
    set_seed(args.seed)
    rng = random.Random(args.seed)
    device = pick_device()

    output = resolve_output(args)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output} is not empty; choose another --output")
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "train_log.jsonl"

    val_limit = max(8, args.limit // 2) if args.limit else None
    train_data = load_split(args.dataset, "train", args.limit)
    val_data = load_split(args.dataset, "validation", val_limit)
    print(f"device: {device} | train: {len(train_data[0])} | validation: {len(val_data[0])}")
    print(f"output: {output}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(device)

    steps_per_epoch = -(-len(train_data[0]) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(0.05 * total_steps), total_steps
    )

    settings = {
        "base_model": BASE_MODEL,
        "seed": args.seed,
        "device": device,
        "limit": args.limit,
        "train_examples": len(train_data[0]),
        "validation_examples": len(val_data[0]),
        "epochs_max": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "max_target": args.max_target,
        "learning_rate": args.lr,
        "patience": args.patience,
        "test_split_used": False,
    }

    def write_summary(history, best_epoch, best_loss, started, step_seconds):
        summary = dict(settings)
        summary.update(
            {
                "best_epoch": best_epoch,
                "best_val_loss": best_loss if best_epoch else None,
                "total_minutes": (time.time() - started) / 60,
                "mean_seconds_per_step": (
                    sum(step_seconds) / len(step_seconds) if step_seconds else None
                ),
                "history": history,
            }
        )
        (output / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    history, best_loss, best_epoch, bad_epochs = [], float("inf"), 0, 0
    step_seconds = []
    global_step = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start, running, seen = time.time(), 0.0, 0
        for step, (inputs, labels) in enumerate(
            make_batches(
                *train_data, tokenizer, args.batch_size, args.max_length,
                args.max_target, True, rng,
            ),
            start=1,
        ):
            step_start = time.time()
            inputs = inputs.to(device)
            loss = model(**inputs, labels=labels.to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            value = loss.item()  # also forces the step to finish
            sync(device)
            step_seconds.append(time.time() - step_start)
            global_step += 1
            running += value * labels.shape[0]
            seen += labels.shape[0]
            if step % args.log_every == 0 or step == steps_per_epoch:
                recent = step_seconds[-args.log_every:]
                sec = sum(recent) / len(recent)
                record = {
                    "type": "step",
                    "epoch": epoch,
                    "step": step,
                    "global_step": global_step,
                    "train_loss_running": running / seen,
                    "sec_per_step": sec,
                }
                append_log(log_path, record)
                print(
                    f"  epoch {epoch} step {step}/{steps_per_epoch} | "
                    f"loss {running / seen:.3f} | {sec:.2f} s/step | "
                    f"~{sec * steps_per_epoch / 60:.1f} min/epoch"
                )

        train_loss = running / seen
        val_loss = evaluate_loss(model, val_data, tokenizer, args, device, rng)
        minutes = (time.time() - epoch_start) / 60
        print(f"epoch {epoch}: train {train_loss:.3f} | val {val_loss:.3f} | {minutes:.1f} min")
        entry = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "minutes": minutes}
        history.append(entry)
        append_log(log_path, {"type": "epoch", **entry})

        if val_loss < best_loss:
            best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            model.save_pretrained(output / "best")
            tokenizer.save_pretrained(output / "best")
            print("  saved best model")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stopping (no improvement for {args.patience} epochs)")
                write_summary(history, best_epoch, best_loss, started, step_seconds)
                break
        write_summary(history, best_epoch, best_loss, started, step_seconds)

    print(f"done. best epoch {best_epoch}, val loss {best_loss:.3f}, saved in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "data/processed/summarization_dataset.csv"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="checkpoint folder, must be empty (default: checkpoints/codet5_summarizer; "
        "with --limit: a new checkpoints/codet5_smoke_<time>)",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-target", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="use only the first N train examples (quick smoke test)",
    )
    parser.add_argument("--log-every", type=int, default=50, help="log every N steps")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
