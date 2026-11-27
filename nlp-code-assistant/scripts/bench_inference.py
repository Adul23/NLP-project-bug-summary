"""Inference speed of the CodeBERT LoRA classifier in different modes (run on a GPU).

Every combination of adapter vs merged LoRA weights, fp32 vs fp16 autocast and eager vs
torch.compile is measured on validation functions: median latency of one forward pass at
batch 1 and batch 16 (tokenization excluded, after warm-up, timed with
torch.cuda.synchronize), peak GPU memory, and agreement with the reference mode
(adapter, fp32, eager) on N functions: share of identical labels and the largest
difference in bug probability.

    python scripts/bench_inference.py --model-dir models/codebert_lora_seed42 --device cuda

Writes a table to the console and results/bench_inference.json.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WEIGHTS = ("adapter", "merged")
PRECISIONS = ("fp32", "fp16")
MODES = ("eager", "compile")
COLUMNS = (
    ("weights", "weights", "{}"),
    ("precision", "prec", "{}"),
    ("mode", "mode", "{}"),
    ("batch1_ms", "b1 ms", "{:.2f}"),
    ("batch16_ms", "b16 ms", "{:.2f}"),
    ("peak_mem_mb", "peak MB", "{:.0f}"),
    ("label_agreement", "agree", "{:.3f}"),
    ("max_abs_prob_diff", "max dp", "{:.1e}"),
)


def median_ms(fn, sync, warmup, repeats):
    """Median wall time of fn() in milliseconds; sync() waits for queued GPU work."""
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def agreement(reference, probs, threshold):
    """Share of identical labels and the largest probability difference."""
    if len(reference) != len(probs) or not probs:
        raise ValueError("need the same non-empty set of predictions")
    same = sum((a >= threshold) == (b >= threshold) for a, b in zip(reference, probs))
    return {
        "label_agreement": same / len(probs),
        "max_abs_prob_diff": max(abs(a - b) for a, b in zip(reference, probs)),
    }


def table(rows):
    """Plain-text table; a failed mode shows its error instead of numbers."""
    header = [title for _, title, _ in COLUMNS]
    lines = []
    for row in rows:
        cells = []
        for key, _, fmt in COLUMNS:
            value = row.get(key)
            cells.append("-" if value is None else fmt.format(value))
        if row.get("error"):
            cells = cells[:3] + ["error: " + row["error"][:60]]
        lines.append(cells)
    widths = [max(len(c[i]) for c in [header] + lines if i < len(c))
              for i in range(len(header))]
    fmt_line = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()  # noqa: E731
    return "\n".join([fmt_line(header), fmt_line(["-" * w for w in widths])]
                     + [fmt_line(c) for c in lines])


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-dir", type=Path, default=ROOT / "models/codebert_lora_seed42")
    p.add_argument("--split-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n", type=int, default=200, help="validation functions for agreement")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--skip-compile", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT / "results/bench_inference.json")
    a = p.parse_args()

    import torch
    from src.bug_detection.experiment_data import DEFAULT_SPLITS, load_splits
    from src.inference.models import LoraPredictor

    cuda = torch.device(a.device).type == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    (_, validation, _), _ = load_splits(a.split_dir or DEFAULT_SPLITS)
    codes = validation.code.sample(n=min(a.n, len(validation)), random_state=42).tolist()
    modes = MODES[:1] if a.skip_compile else MODES

    rows, reference, threshold = [], None, 0.5
    for weights in WEIGHTS:
        predictor = LoraPredictor(a.model_dir, device=a.device, merge=weights == "merged")
        threshold = predictor.config.get("decision_threshold") or 0.5
        tokenizer, eager = predictor.tokenizer, predictor.model

        def encode(batch):
            return tokenizer(batch, return_tensors="pt", truncation=True, padding=True,
                             max_length=predictor.config["max_length"],
                             return_token_type_ids=False).to(a.device)

        for mode in modes:
            model = torch.compile(eager) if mode == "compile" else eager
            for precision in PRECISIONS:
                row = {"weights": weights, "precision": precision, "mode": mode}
                print(f"{weights} {precision} {mode} ...", flush=True)

                def forward(inputs):
                    with torch.inference_mode(), torch.autocast(
                        device_type="cuda" if cuda else "cpu",
                        dtype=torch.float16, enabled=precision == "fp16",
                    ):
                        return model(**inputs).logits.float().softmax(-1)[:, 1]

                try:
                    if cuda:
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                    probs = []
                    for start in range(0, len(codes), 16):
                        probs += forward(encode(codes[start:start + 16])).tolist()
                    if reference is None:
                        reference = probs  # adapter, fp32, eager comes first
                    row.update(agreement(reference, probs, threshold))
                    for size in (1, 16):
                        inputs = encode(codes[:size])
                        row[f"batch{size}_ms"] = median_ms(
                            lambda: forward(inputs), sync, a.warmup, a.repeats)
                    if cuda:
                        row["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
                except Exception as error:  # e.g. torch.compile unsupported here
                    row["error"] = f"{type(error).__name__}: {error}"
                rows.append(row)
        del predictor, eager, model
        if cuda:
            torch.cuda.empty_cache()

    print()
    print(table(rows))
    result = {
        "model_dir": str(a.model_dir),
        "device": torch.cuda.get_device_name() if cuda else a.device,
        "torch": torch.__version__,
        "n_agreement": len(codes),
        "threshold": threshold,
        "warmup": a.warmup,
        "repeats": a.repeats,
        "reference": "adapter fp32 eager",
        "rows": rows,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwritten {a.output}")


if __name__ == "__main__":
    main()
