"""Top-1 / top-3 accuracy of operator-flip bug localization on test mutants.

For every buggy test function, the original (same source_id, label 0) is the true fix.
localize() ranks single-operator repairs by the drop in bug probability; we record where
the true fix lands. The random baseline is what picking a candidate uniformly would give.

    python -m src.evaluation.localization_eval --model codebert --device cuda

With --data <csv> (e.g. real bug pairs from scripts/collect_real_bugs.py) the CSV replaces
the test split, results go to results/real_bugs/, and the classifier itself is also scored
on every row (F1 / precision / recall, overall and by the optional `in_tables` column).
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
    has_in_tables = "in_tables" in buggy.columns
    rows = []
    skipped = 0
    for i, row in enumerate(buggy.itertuples(index=False), start=1):
        original = originals[row.source_id]
        try:
            result = localize(row.code, predictor, top_k=None)  # full ranking
        except SyntaxError:  # only possible with external data (--data)
            skipped += 1
            continue
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
                **({"in_tables": bool(row.in_tables)} if has_in_tables else {}),
            }
        )
        if progress and i % 100 == 0:
            print(f"{i}/{len(buggy)}")
    frame = pd.DataFrame(rows)
    summary = summarize(frame)
    summary['skipped_unparsable'] = skipped
    return frame, summary


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
    summary = {
        "all": _stats(frame),
        "by_mutation_type": {k: _stats(g) for k, g in frame.groupby("mutation_type")},
        "classifier_detected": _stats(frame[frame.detected])
        if frame.detected.any()
        else None,
        "classifier_missed": _stats(frame[~frame.detected])
        if (~frame.detected).any()
        else None,
    }
    if "in_tables" in frame:
        for name, part in (("in_tables", frame[frame.in_tables]),
                           ("not_in_tables", frame[~frame.in_tables])):
            summary[name] = _stats(part) if len(part) else None
    return summary


def score_classifier(data: pd.DataFrame, predictor, threshold: float) -> dict:
    """Classification metrics on every row, overall and split by `in_tables` if present."""
    from src.evaluation.bug_metrics import compute_bug_metrics

    probs = [r["bug_probability"] for r in predictor.predict_batch(data.code.tolist())]
    data = data.assign(prob=probs, pred=[int(p >= threshold) for p in probs])

    def metrics(part):
        return compute_bug_metrics(part.label, part.pred, part.prob, part.mutation_type)

    result = {"threshold": threshold, "all": metrics(data)}
    if "in_tables" in data:
        # pairs share in_tables, so both labels are in each part
        for name, part in (("in_tables", data[data.in_tables]),
                           ("not_in_tables", data[~data.in_tables])):
            result[name] = metrics(part) if len(part) else None
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=["tfidf", "codebert"], default="codebert")
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--split-dir", type=Path, default=DEFAULT_SPLITS)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--limit", type=int, default=None, help="first N buggy functions (smoke test)"
    )
    p.add_argument(
        "--data", type=Path, default=None,
        help="CSV with sample_id, source_id, code, label, mutation_type (replaces the test split)",
    )
    p.add_argument("--output", type=Path, default=None,
                   help="default: results/localization, or results/real_bugs with --data")
    a = p.parse_args()
    output = a.output or ROOT / ("results/real_bugs" if a.data else "results/localization")

    from src.inference.models import LoraPredictor, TfidfPredictor

    if a.model == "tfidf":
        predictor = TfidfPredictor(a.model_dir or ROOT / "models/tfidf_v2")
        threshold = 0.5
    else:
        predictor = LoraPredictor(
            a.model_dir or ROOT / "models/codebert_lora", device=a.device
        )
        threshold = predictor.config.get("decision_threshold") or 0.5

    if a.data:
        test = pd.read_csv(a.data, dtype={"source_id": str})
    else:
        (_, _, test), _ = load_splits(a.split_dir)
    if a.limit:
        keep = test[test.label == 1].source_id.head(a.limit)
        test = test[test.source_id.isin(keep)]
    rows, summary = evaluate(test, predictor, threshold)
    output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output / f"{a.model}_rows.csv", index=False)
    (output / f"{a.model}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["all"], indent=2))
    if a.data:
        scores = score_classifier(test, predictor, threshold)
        (output / f"{a.model}_classifier.json").write_text(json.dumps(scores, indent=2) + "\n")
        for name in ("all", "in_tables", "not_in_tables"):
            m = scores.get(name)
            if m:
                print(f"classifier {name:13s} n={m['n']:4d}  F1 {m['bug_f1']:.3f}  "
                      f"P {m['bug_precision']:.3f}  R {m['bug_recall']:.3f}  acc {m['accuracy']:.3f}")


if __name__ == "__main__":
    main()
