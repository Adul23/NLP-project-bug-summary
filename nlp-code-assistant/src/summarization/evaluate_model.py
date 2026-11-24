"""Generate summaries with a fine-tuned CodeT5 checkpoint and score them on one split.

Works with a full model folder or a LoRA adapter folder (detected by adapter_config.json,
or pass --lora-adapter PATH). Uses the same dataset, split column and metrics (metrics.py)
as the baselines in evaluate_baseline.py, so the numbers are directly comparable.
Also reports single-example latency (batch 1, warmed-up model) and artifact size.
Default split is validation. The test split needs --confirm-final-test (final evaluation only).
Only local files are loaded; nothing is downloaded.

Smoke check:   python -m src.summarization.evaluate_model --limit 32
Validation:    python -m src.summarization.evaluate_model
LoRA:          python -m src.summarization.evaluate_model --lora-adapter checkpoints/codet5_summarizer_lora/best
Comparison:    python -m src.summarization.evaluate_model --compare
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from pathlib import Path

import pandas as pd
import torch

from src.summarization import final_test_guard
from src.summarization.metrics import corpus_bleu, mean_rouge_l, rouge_l_f1, tokenize
from src.summarization.model_loading import (
    artifact_size_mb,
    is_lora_checkpoint,
    load_model_and_tokenizer,
)
from src.summarization.train_codet5 import pick_device, sync

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data/processed/summarization_dataset.csv"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints/codet5_summarizer/best"
DEFAULT_LORA_CHECKPOINT = PROJECT_ROOT / "checkpoints/codet5_summarizer_lora/best"
DEFAULT_COMPARE_DIR = PROJECT_ROOT / "results_summarization"  # not results/ (teammate's)


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


def measure_latency(
    model, tokenizer, codes, device, max_length, max_new_tokens, num_beams,
    generation_kwargs=None, count=100, warmup=3,
):
    """Wall-clock time of one generate() call on a single example (batch 1), in ms.

    The first `warmup` calls are not timed (kernel compilation, MPS/CUDA warm-up).
    """
    sample = codes[:count]
    times = []
    model.eval()

    def one(code):
        encoded = tokenizer(
            code, max_length=max_length, truncation=True, return_tensors="pt"
        ).to(device)
        model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
            **(generation_kwargs or {}),
        )

    with torch.inference_mode():
        for code in sample[:warmup]:
            one(code)
        sync(device)
        for code in sample:
            started = time.perf_counter()
            one(code)
            sync(device)
            times.append((time.perf_counter() - started) * 1000)
    return {
        "latency_ms_median": round(statistics.median(times), 1),
        "latency_ms_mean": round(statistics.fmean(times), 1),
        "latency_examples": len(times),
    }


def generation_options(args):
    # Only options that were given are passed, so a run without them is the original setup.
    options = {}
    if args.length_penalty is not None:
        options["length_penalty"] = args.length_penalty
    if args.min_new_tokens is not None:
        options["min_new_tokens"] = args.min_new_tokens
    return options


def load_subset(args):
    df = pd.read_csv(args.dataset, dtype={"source_id": str}, keep_default_na=False)
    subset = df[df["split"] == args.split]
    if args.limit:
        subset = subset.head(args.limit)
    if subset.empty:
        raise ValueError(f"No rows for split {args.split!r}")
    return subset


def file_suffix(args, extra=""):
    suffix = f"_limit{args.limit}" if args.limit else ""
    if args.tag:
        suffix += f"_{args.tag}"
    return suffix + extra


def evaluate_checkpoint(args, checkpoint, subset, predictions_path, metrics_path, device):
    """Score one checkpoint (full or LoRA) on `subset` and write predictions + metrics."""
    options = generation_options(args)
    model, tokenizer, kind = load_model_and_tokenizer(
        checkpoint, device, merge_lora=args.merge_lora
    )
    codes = subset["code"].tolist()
    references = subset["summary"].tolist()

    started = time.time()
    hypotheses = generate_summaries(
        model, tokenizer, codes, device,
        args.batch_size, args.max_length, args.max_new_tokens, args.num_beams, options,
    )
    seconds = time.time() - started
    latency = measure_latency(
        model, tokenizer, codes, device, args.max_length, args.max_new_tokens,
        args.num_beams, options, count=args.latency_examples,
    )

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
    if kind == "lora":
        trainable = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    else:
        trainable = sum(p.numel() for p in model.parameters())
    result = {
        "method": "codet5_lora" if kind == "lora" else "codet5_finetuned",
        "split": args.split,
        "examples": len(subset),
        "bleu4": round(corpus_bleu(references, hypotheses), 4),
        "rouge_l_f1": round(mean_rouge_l(references, hypotheses), 4),
        "exact_match": round(exact / len(subset), 4),
        "empty_predictions": sum(not h for h in hypotheses),
        "mean_words_reference": round(sum(len(tokenize(r)) for r in references) / len(subset), 2),
        "mean_words_prediction": round(sum(len(tokenize(h)) for h in hypotheses) / len(subset), 2),
        "checkpoint": str(checkpoint),
        "brevity_penalty": round(brevity_penalty, 4),
        "tag": args.tag,
        "num_beams": args.num_beams,
        "length_penalty": args.length_penalty,
        "min_new_tokens": args.min_new_tokens,
        "max_new_tokens": args.max_new_tokens,
        "max_length": args.max_length,
        "generation_seconds": round(seconds, 1),
        "device": device,
        "lora_merged": bool(kind == "lora" and args.merge_lora),
        "trainable_parameters": trainable,
        "artifact_size_mb": artifact_size_mb(checkpoint),
        **latency,
    }
    metrics_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"saved: {predictions_path}\nsaved: {metrics_path}")
    del model
    return result


def refuse_existing(*paths):
    for path in paths:
        if path.exists():
            raise ValueError(f"{path} already exists; choose another --output-dir")


def resolve_checkpoint(args):
    if args.lora_adapter is not None:
        if not is_lora_checkpoint(args.lora_adapter):
            raise ValueError(f"{args.lora_adapter} has no adapter_config.json (not a LoRA adapter)")
        return args.lora_adapter
    return args.checkpoint  # full model, or an adapter folder (auto-detected)


def evaluate(args: argparse.Namespace) -> dict:
    subset = load_subset(args)
    checkpoint = resolve_checkpoint(args)
    suffix = file_suffix(args)
    output_dir = args.output_dir or checkpoint.parent
    predictions_path = output_dir / f"predictions_{args.split}{suffix}.csv"
    metrics_path = output_dir / f"metrics_{args.split}{suffix}.json"
    refuse_existing(predictions_path, metrics_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(
        f"device: {device} | split: {args.split} | examples: {len(subset)} "
        f"| beams: {args.num_beams} | checkpoint: {checkpoint}"
    )
    return evaluate_checkpoint(args, checkpoint, subset, predictions_path, metrics_path, device)


# ---------------------------------------------------------------- comparison table

BASELINE_ROWS = (
    ("ast", "AST template"),
    ("tfidf", "TF-IDF retrieval"),
    ("bm25", "BM25 retrieval"),
)
COLUMNS = (
    "Method", "BLEU-4", "ROUGE-L F1", "Latency median (ms)", "Latency mean (ms)",
    "Artifact (MB)", "Trainable params",
)


def _cell(value, spec=None):
    if value is None:
        return "—"
    if value == "TBD":
        return "TBD"
    return format(value, spec) if spec else str(value)


def render_comparison(rows: list[dict], split: str, examples: int) -> str:
    """Markdown table. A missing value is '—' (not applicable), 'TBD' (not run yet)."""
    lines = [
        f"Split: {split} ({examples} examples). Same dataset, split and metrics for every row.",
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "---|" * len(COLUMNS),
    ]
    for r in rows:
        lines.append(
            "| " + " | ".join(
                [
                    r["label"],
                    _cell(r.get("bleu4"), ".4f"),
                    _cell(r.get("rouge_l_f1"), ".4f"),
                    _cell(r.get("latency_ms_median"), ".1f"),
                    _cell(r.get("latency_ms_mean"), ".1f"),
                    _cell(r.get("artifact_size_mb"), ".2f"),
                    _cell(r.get("trainable_parameters"), ","),
                ]
            ) + " |"
        )
    lines += [
        "",
        "Latency: one example per call (batch 1), warmed-up model, beam search as set for "
        "the run. '—' = not measured or not applicable (baselines have no neural model); "
        "'TBD' = checkpoint not available yet.",
    ]
    return "\n".join(lines) + "\n"


def compare(args: argparse.Namespace) -> list[dict]:
    from src.summarization import evaluate_baseline

    subset = load_subset(args)
    output_dir = args.output_dir or DEFAULT_COMPARE_DIR
    suffix = file_suffix(args)
    json_path = output_dir / f"comparison_{args.split}{suffix}.json"
    md_path = output_dir / f"comparison_{args.split}{suffix}.md"
    full_pred = output_dir / f"predictions_{args.split}{suffix}_full.csv"
    lora_pred = output_dir / f"predictions_{args.split}{suffix}_lora.csv"
    full_metrics = output_dir / f"metrics_{args.split}{suffix}_full.json"
    lora_metrics = output_dir / f"metrics_{args.split}{suffix}_lora.json"
    refuse_existing(json_path, md_path, full_pred, lora_pred, full_metrics, lora_metrics)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for method, label in BASELINE_ROWS:
        result = evaluate_baseline.evaluate(args.dataset, args.split, method, limit=args.limit)
        rows.append({**result, "label": label, "trainable_parameters": None})
        print(f"{label}: BLEU-4 {result['bleu4']} | ROUGE-L {result['rouge_l_f1']}")

    device = pick_device()
    lora_path = args.lora_adapter or DEFAULT_LORA_CHECKPOINT
    for label, path, preds, metrics in (
        ("CodeT5-small full FT", args.checkpoint, full_pred, full_metrics),
        ("CodeT5-small LoRA", lora_path, lora_pred, lora_metrics),
    ):
        if not (Path(path) / "config.json").is_file() and not is_lora_checkpoint(path):
            print(f"{label}: no checkpoint in {path} -> TBD")
            rows.append({k: "TBD" for k in ("bleu4", "rouge_l_f1", "latency_ms_median",
                                             "latency_ms_mean", "artifact_size_mb",
                                             "trainable_parameters")} | {"label": label})
            continue
        print(f"{label}: evaluating {path} on {device}")
        result = evaluate_checkpoint(args, Path(path), subset, preds, metrics, device)
        rows.append({**result, "label": label})

    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    md_path.write_text(render_comparison(rows, args.split, len(subset)))
    print(f"saved: {json_path}\nsaved: {md_path}")
    print(md_path.read_text())
    return rows


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="full model folder (a LoRA adapter folder is detected automatically)")
    parser.add_argument("--lora-adapter", type=Path, default=None,
                        help="LoRA adapter folder (overrides --checkpoint); "
                        f"in --compare the default is {DEFAULT_LORA_CHECKPOINT.relative_to(PROJECT_ROOT)}")
    parser.add_argument("--merge-lora", action="store_true",
                        help="merge the adapter into the base weights before generating "
                        "(faster; default: adapter kept separate)")
    parser.add_argument("--compare", action="store_true",
                        help="one table: AST, TF-IDF, BM25, CodeT5 full FT, CodeT5 LoRA")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument(
        "--confirm-final-test", "--final-test", dest="confirm_final_test", action="store_true",
        help="required for --split test: the one-time final evaluation",
    )
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: the folder that contains the checkpoint "
                        f"(--compare: {DEFAULT_COMPARE_DIR.name}/)")
    parser.add_argument("--limit", type=int, default=None, help="first N examples only (smoke check)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=384, help="input tokens, same as training")
    parser.add_argument("--max-new-tokens", type=int, default=48, help="same as max_target in training")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=None,
                        help="beam search length penalty (>1 favors longer outputs); default: model default")
    parser.add_argument("--min-new-tokens", type=int, default=None,
                        help="minimum generated tokens; default: no minimum")
    parser.add_argument("--latency-examples", type=int, default=100,
                        help="examples timed one by one (batch 1) for the latency columns")
    parser.add_argument("--tag", default=None,
                        help="label added to the output file names, e.g. A (letters, digits, - and _)")
    args = parser.parse_args(argv)
    if args.tag and not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        parser.error("--tag may contain only letters, digits, '-' and '_'")
    final_test_guard.check_args(parser, args.split, args.confirm_final_test)
    return args


def main() -> None:
    args = parse_args()
    if args.compare:
        compare(args)
        return
    result = evaluate(args)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
