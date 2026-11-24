"""Top-1 / top-3 accuracy of operator-flip bug localization on test mutants.

For every buggy test function, the original (same source_id, label 0) is the true fix.
localize() ranks single-operator repairs by the drop in bug probability; we record where
the true fix lands. The random baseline is what picking a candidate uniformly would give.

    python -m src.evaluation.localization_eval --model codebert --device cuda
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from src.bug_detection.experiment_data import DEFAULT_SPLITS, load_splits
from src.bug_detection.localize import _changed_line, localize
from src.bug_detection.split_dataset import ROOT


def evaluate(
    test: pd.DataFrame, predictor, threshold: float = 0.5, progress: bool = True
):
    """Rows (one per buggy function with an original) and a summary dict."""
    originals = test[test.label == 0].set_index("source_id").code
    buggy = test[(test.label == 1) & test.source_id.isin(originals.index)]
    rows = []
    for i, row in enumerate(buggy.itertuples(index=False), start=1):
        original = originals[row.source_id]
        result = localize(row.code, predictor, top_k=None)  # full ranking
        codes = [c["code"] for c in result["candidates"]]
        rank = codes.index(original) + 1 if original in codes else None
        true_line = _changed_line(row.code, original)
        best = result["candidates"][0] if codes else None
        rows.append(
            {
                "sample_id": row.sample_id,
                "mutation_type": row.mutation_type,
                "n_candidates": result["n_candidates"],
                "rank": rank,
                "top1": rank == 1,
                "top3": rank is not None and rank <= 3,
                "line_top1": best is not None and best["line"] == true_line,
                "detected": result["bug_probability"] >= threshold,
                "bug_probability": result["bug_probability"],
            }
        )
        if progress and i % 100 == 0:
            print(f"{i}/{len(buggy)}")
    frame = pd.DataFrame(rows)
    return frame, summarize(frame)


def _stats(frame: pd.DataFrame) -> dict:
    n = frame.n_candidates
    return {
        "n": len(frame),
        "top1": float(frame.top1.mean()),
        "top3": float(frame.top3.mean()),
        "line_top1": float(frame.line_top1.mean()),
        "true_fix_missing": int(frame["rank"].isna().sum()),
        "mean_candidates": float(n.mean()),
        "random_top1": float((1 / n.clip(lower=1)).mean()),
        "random_top3": float((n.clip(upper=3) / n.clip(lower=1)).mean()),
    }


def summarize(frame: pd.DataFrame) -> dict:
    return {
        "all": _stats(frame),
        "by_mutation_type": {k: _stats(g) for k, g in frame.groupby("mutation_type")},
        "classifier_detected": _stats(frame[frame.detected])
        if frame.detected.any()
        else None,
        "classifier_missed": _stats(frame[~frame.detected])
        if (~frame.detected).any()
        else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=["tfidf", "codebert"], default="codebert")
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--split-dir", type=Path, default=DEFAULT_SPLITS)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--limit", type=int, default=None, help="first N buggy functions (smoke test)"
    )
    p.add_argument("--output", type=Path, default=ROOT / "results/localization")
    a = p.parse_args()

    from src.inference.models import LoraPredictor, TfidfPredictor

    if a.model == "tfidf":
        predictor = TfidfPredictor(a.model_dir or ROOT / "models/tfidf_v2")
        threshold = 0.5
    else:
        predictor = LoraPredictor(
            a.model_dir or ROOT / "models/codebert_lora", device=a.device
        )
        threshold = predictor.config.get("decision_threshold") or 0.5

    (_, _, test), _ = load_splits(a.split_dir)
    if a.limit:
        keep = test[test.label == 1].source_id.head(a.limit)
        test = test[test.source_id.isin(keep)]
    rows, summary = evaluate(test, predictor, threshold)
    a.output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(a.output / f"{a.model}_rows.csv", index=False)
    (a.output / f"{a.model}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary["all"], indent=2))


if __name__ == "__main__":
    main()
