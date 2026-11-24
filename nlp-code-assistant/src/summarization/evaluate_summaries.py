"""Score CodeT5 and every summarization baseline on the same rows with the full metric set.

    python -m src.summarization.evaluate_summaries                 # test split, all methods
    python -m src.summarization.evaluate_summaries --limit 200     # quick run

Baselines: ast (template), tfidf and bm25 (retrieve the summary of the most similar TRAIN
function). CodeT5 is the fully fine-tuned model in models/codet5_summarizer (loaded locally,
never downloaded). The split is only scored; nothing is tuned on it.

Writes results/summarization_metrics.json, per-row outputs in
results/summarization_<split>_predictions.csv and, for the test split, a blank
human-evaluation sheet results/summarization_human_eval.csv (50 random rows).
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import pandas as pd

from src.summarization import ast_baseline, bm25_retrieval, tfidf_retrieval
from src.summarization.metrics import summary_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data/processed/summarization_dataset.csv"
METHODS = ("ast", "tfidf", "bm25", "codet5")
METHOD_NAMES = {
    "ast": "AST template",
    "tfidf": "TF-IDF retrieval",
    "bm25": "BM25 retrieval",
    "codet5": "CodeT5-small (fine-tuned)",
}
HUMAN_EVAL_COLUMNS = ["correctness_0_2", "fluency_0_2", "specificity_0_2"]
GENERATION = {"num_beams": 4, "max_new_tokens": 48, "no_repeat_ngram_size": 3, "early_stopping": True}


def load_rows(dataset: Path, split: str, limit: int | None):
    if not Path(dataset).exists():
        raise FileNotFoundError(
            f"{dataset} not found. Build it with src/summarization/build_summary_dataset.py and "
            "src/summarization/apply_shared_split.py, or copy the file used for training."
        )
    frame = pd.read_csv(dataset, dtype={"source_id": str}, keep_default_na=False)
    rows = frame[frame["split"] == split]
    if limit:
        rows = rows.head(limit)
    train = frame[frame["split"] == "train"]
    if rows.empty:
        raise ValueError(f"No rows for split {split!r}")
    return rows.reset_index(drop=True), train


def codet5_generate(codes, model_dir, device="auto", batch_size=32, latency_samples=20, summarizer=None):
    """Batched beam-search summaries plus batch-1 latency on a few rows."""
    import torch

    from src.inference.models import CodeSummarizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = summarizer or CodeSummarizer(model_dir, device=device)
    tokenizer, network = model.tokenizer, model.model
    settings = {**GENERATION, "max_new_tokens": model.max_target}
    order = sorted(range(len(codes)), key=lambda i: len(codes[i]), reverse=True)
    outputs = [""] * len(codes)
    started = time.perf_counter()
    for start in range(0, len(order), batch_size):
        rows = order[start : start + batch_size]
        inputs = tokenizer(
            [codes[i] for i in rows], return_tensors="pt", padding=True, truncation=True, max_length=model.max_length
        ).to(network.device)
        with torch.inference_mode():
            ids = network.generate(**inputs, **settings)
        for i, text in zip(rows, tokenizer.batch_decode(ids, skip_special_tokens=True)):
            outputs[i] = text.strip()
        if (start // batch_size) % 10 == 0:
            print(f"  codet5 {min(start + batch_size, len(order))}/{len(order)} rows", flush=True)
    batched_seconds = time.perf_counter() - started

    timings = []
    for code in codes[: min(latency_samples, len(codes))]:
        inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=model.max_length).to(network.device)
        began = time.perf_counter()
        with torch.inference_mode():
            network.generate(**inputs, **settings)
        timings.append((time.perf_counter() - began) * 1000)
    efficiency = {
        "device": str(network.device),
        "generation": settings,
        "batch_size": batch_size,
        "batched_ms_per_function": batched_seconds * 1000 / len(codes),
        "latency_ms_p50_batch1": statistics.median(timings) if timings else None,
        "latency_samples": len(timings),
    }
    return outputs, efficiency


def evaluate(
    dataset=DEFAULT_DATASET,
    split="test",
    methods=METHODS,
    *,
    model_dir=PROJECT_ROOT / "models/codet5_summarizer",
    output=PROJECT_ROOT / "results/summarization_metrics.json",
    human_eval=PROJECT_ROOT / "results/summarization_human_eval.csv",
    human_eval_rows=50,
    limit=None,
    device="auto",
    batch_size=32,
    n_bootstrap=1000,
    summarizer=None,
):
    rows, train = load_rows(dataset, split, limit)
    codes, references = rows["code"].tolist(), rows["summary"].tolist()
    outputs, systems, notes = {}, {}, []
    for method in methods:
        print(f"{METHOD_NAMES[method]} on {len(rows)} {split} rows", flush=True)
        extra = {}
        started = time.perf_counter()
        if method == "ast":
            hypotheses = [ast_baseline.predict(code) for code in codes]
        elif method in ("tfidf", "bm25"):
            if train.empty:
                raise ValueError("No train rows to build the retrieval index")
            retrieve = (tfidf_retrieval if method == "tfidf" else bm25_retrieval).retrieve_summaries
            hypotheses, no_overlap = retrieve(train["code"].tolist(), train["summary"].tolist(), codes)
            extra["queries_without_token_overlap"] = no_overlap
        else:
            try:
                hypotheses, extra["efficiency"] = codet5_generate(
                    codes, model_dir, device, batch_size, summarizer=summarizer
                )
            except (OSError, ImportError) as exc:
                notes.append(f"CodeT5 skipped: {exc}")
                continue
        if method != "codet5":
            extra["ms_per_function"] = (time.perf_counter() - started) * 1000 / len(codes)
        outputs[method] = hypotheses
        systems[method] = {
            "name": METHOD_NAMES[method],
            **summary_metrics(references, hypotheses, codes, n_bootstrap=n_bootstrap),
            **extra,
        }

    predictions = rows[["source_id", "code", "summary"]].rename(columns={"summary": "reference"})
    for method, hypotheses in outputs.items():
        predictions[method] = hypotheses
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = output.parent / f"summarization_{split}_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    if split == "test" and human_eval:
        sample = predictions.sample(n=min(human_eval_rows, len(predictions)), random_state=42)
        sheet = sample[["source_id", "code", "reference"] + (["codet5"] if "codet5" in outputs else [])]
        sheet = sheet.rename(columns={"codet5": "generated"})
        for column in HUMAN_EVAL_COLUMNS:
            sheet[column] = ""  # filled in by human raters, never automatically
        sheet["notes"] = ""
        sheet.to_csv(human_eval, index=False)

    result = {
        "split": split,
        "examples": len(rows),
        "limit": limit,
        "dataset": str(dataset),
        "systems": systems,
        "protocol": {
            "tokenization": "lowercase word/number tokens (src/summarization/metrics.py)",
            "scale": "all scores in [0, 1]",
            "bleu4": "corpus BLEU-4, uniform weights, brevity penalty",
            "sentence_bleu4_smoothed": "mean sentence BLEU-4 with Lin & Och add-one smoothing",
            "rouge": "mean per-row ROUGE-1/2/L precision, recall, F1",
            "chrf": "corpus chrF, character n-grams 1-6, beta 2, whitespace ignored",
            "bootstrap": f"95% percentile CI, {n_bootstrap} resamples, seed 42 (BLEU-4, ROUGE-L F1)",
            "retrieval_index": "train split only",
            "codet5_generation": GENERATION,
        },
        "predictions_file": predictions_path.name,
        "human_eval_file": Path(human_eval).name if split == "test" and human_eval else None,
        "machine": platform.platform(),
        "notes": notes,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models/codet5_summarizer")
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/summarization_metrics.json")
    p.add_argument("--human-eval", type=Path, default=PROJECT_ROOT / "results/summarization_human_eval.csv")
    p.add_argument("--limit", type=int, default=None, help="score only the first N rows (quick run)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--bootstrap", type=int, default=1000)
    a = p.parse_args()
    result = evaluate(
        a.dataset,
        a.split,
        tuple(a.methods),
        model_dir=a.model_dir,
        output=a.output,
        human_eval=a.human_eval,
        limit=a.limit,
        device=a.device,
        batch_size=a.batch_size,
        n_bootstrap=a.bootstrap,
    )
    for method, system in result["systems"].items():
        print(
            f"{system['name']:<28} BLEU-4 {system['bleu4']:.4f}  ROUGE-L {system['rouge_l']['f1']:.4f}  "
            f"chrF {system['chrf']:.4f}"
        )
    for note in result["notes"]:
        print("NOTE:", note)


if __name__ == "__main__":
    main()
