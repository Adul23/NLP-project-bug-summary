"""Generate summaries with the fine-tuned CodeT5 checkpoint and score them on one split.

Uses the same dataset, split column and metrics (metrics.py) as the baselines in
evaluate_baseline.py, so the numbers are directly comparable.
Default split is validation. The test split needs --final-test (final evaluation only).
Only local files are loaded; nothing is downloaded.

Smoke check:  python -m src.summarization.evaluate_model --limit 32
Validation:   python -m src.summarization.evaluate_model
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.summarization.metrics import corpus_bleu, mean_rouge_l, rouge_l_f1, tokenize
from src.summarization.train_codet5 import pick_device

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data/processed/summarization_dataset.csv"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints/codet5_summarizer/best"


def generate_summaries(
    model, tokenizer, codes, device, batch_size, max_length, max_new_tokens, num_beams,
    generation_kwargs=None,
):
    """Greedy/beam generation. Inputs are sorted by length for speed, order is restored."""
    order = sorted(range(len(codes)), key=lambda i: len(codes[i]))
    result = [""] * len(codes)
    model.eval()
    started = time.time()
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            encoded = tokenizer(
                [codes[i] for i in idx],
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            ids = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
                **(generation_kwargs or {}),
            )
            for i, text in zip(idx, tokenizer.batch_decode(ids, skip_special_tokens=True)):
                result[i] = text.strip()
            done = min(start + batch_size, len(order))
            if (start // batch_size) % 10 == 0 or done == len(order):
                print(f"  generated {done}/{len(order)} | {time.time() - started:.0f}s")
    return result


def evaluate(args: argparse.Namespace) -> dict:
    df = pd.read_csv(args.dataset, dtype={"source_id": str}, keep_default_na=False)
    subset = df[df["split"] == args.split]
    if args.limit:
        subset = subset.head(args.limit)
    if subset.empty:
        raise ValueError(f"No rows for split {args.split!r}")

    suffix = f"_limit{args.limit}" if args.limit else ""
    if args.tag:
        suffix += f"_{args.tag}"
    # Only options that were given are passed, so a run without them is the original setup.
    generation_kwargs = {}
    if args.length_penalty is not None:
        generation_kwargs["length_penalty"] = args.length_penalty
    if args.min_new_tokens is not None:
        generation_kwargs["min_new_tokens"] = args.min_new_tokens
    output_dir = args.output_dir or args.checkpoint.parent
    predictions_path = output_dir / f"predictions_{args.split}{suffix}.csv"
    metrics_path = output_dir / f"metrics_{args.split}{suffix}.json"
    for path in (predictions_path, metrics_path):
        if path.exists():
            raise ValueError(f"{path} already exists; choose another --output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"device: {device} | split: {args.split} | examples: {len(subset)} | beams: {args.num_beams}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint, local_files_only=True).to(device)

    started = time.time()
    references = subset["summary"].tolist()
    hypotheses = generate_summaries(
        model, tokenizer, subset["code"].tolist(), device,
        args.batch_size, args.max_length, args.max_new_tokens, args.num_beams,
        generation_kwargs,
    )
    seconds = time.time() - started

    scores = [rouge_l_f1(r, h) for r, h in zip(references, hypotheses)]
    pd.DataFrame(
        {
            "source_id": subset["source_id"].tolist(),
            "reference": references,
            "prediction": hypotheses,
            "rouge_l": [round(s, 4) for s in scores],
        }
    ).to_csv(predictions_path, index=False)

    ref_len = sum(len(tokenize(r)) for r in references)
    hyp_len = sum(len(tokenize(h)) for h in hypotheses)
    brevity_penalty = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    exact = sum(tokenize(r) == tokenize(h) for r, h in zip(references, hypotheses))
    result = {
        "method": "codet5_finetuned",
        "split": args.split,
        "examples": len(subset),
        "bleu4": round(corpus_bleu(references, hypotheses), 4),
        "rouge_l_f1": round(mean_rouge_l(references, hypotheses), 4),
        "exact_match": round(exact / len(subset), 4),
        "empty_predictions": sum(not h for h in hypotheses),
        "mean_words_reference": round(sum(len(tokenize(r)) for r in references) / len(subset), 2),
        "mean_words_prediction": round(sum(len(tokenize(h)) for h in hypotheses) / len(subset), 2),
        "checkpoint": str(args.checkpoint),
        "brevity_penalty": round(brevity_penalty, 4),
        "tag": args.tag,
        "num_beams": args.num_beams,
        "length_penalty": args.length_penalty,
        "min_new_tokens": args.min_new_tokens,
        "max_new_tokens": args.max_new_tokens,
        "max_length": args.max_length,
        "generation_seconds": round(seconds, 1),
    }
    metrics_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"saved: {predictions_path}\nsaved: {metrics_path}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument(
        "--final-test", action="store_true",
        help="required for --split test: the one-time final evaluation",
    )
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: the folder that contains the checkpoint")
    parser.add_argument("--limit", type=int, default=None, help="first N examples only (smoke check)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=384, help="input tokens, same as training")
    parser.add_argument("--max-new-tokens", type=int, default=48, help="same as max_target in training")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=None,
                        help="beam search length penalty (>1 favors longer outputs); default: model default")
    parser.add_argument("--min-new-tokens", type=int, default=None,
                        help="minimum generated tokens; default: no minimum")
    parser.add_argument("--tag", default=None,
                        help="label added to the output file names, e.g. A (letters, digits, - and _)")
    args = parser.parse_args()
    if args.tag and not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        parser.error("--tag may contain only letters, digits, '-' and '_'")
    if args.split == "test" and not args.final_test:
        parser.error("the test split is for the final evaluation only; pass --final-test")
    return args


def main() -> None:
    result = evaluate(parse_args())
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
