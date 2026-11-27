"""McNemar test: CodeBERT vs TF-IDF localization on the same test mutants.

Reads results/localization/{codebert,tfidf}_rows.csv (written by localization_eval.py),
checks that both cover the same mutants in the same order, and tests top-1 (and, for
reference, top-3 and correct-line) accuracy with bug_metrics.mcnemar.

    python -m src.evaluation.localization_compare
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from src.bug_detection.split_dataset import ROOT
from src.evaluation.bug_metrics import mcnemar

METRICS = ("top1", "top3", "line_top1")


def compare(a: pd.DataFrame, b: pd.DataFrame, names=("codebert", "tfidf")) -> dict:
    if list(a.sample_id) != list(b.sample_id):
        raise ValueError("row files cover different mutants or a different order")
    result = {"n": len(a), "model_a": names[0], "model_b": names[1]}
    for metric in METRICS:
        hits_a = a[metric].astype(bool).to_numpy()
        hits_b = b[metric].astype(bool).to_numpy()
        # y_true = all ones: a model is "correct" on a mutant when it hit it.
        test = mcnemar(np.ones(len(a), dtype=int), hits_a, hits_b)
        result[metric] = {
            names[0]: float(hits_a.mean()),
            names[1]: float(hits_b.mean()),
            **test,
        }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dir", type=Path, default=ROOT / "results/localization")
    a = p.parse_args()
    rows = {}
    for name in ("codebert", "tfidf"):
        path = a.dir / f"{name}_rows.csv"
        if not path.exists():
            raise SystemExit(
                f"{path} is missing: run localization_eval.py --model {name} first"
            )
        rows[name] = pd.read_csv(path)
    result = compare(rows["codebert"], rows["tfidf"])
    (a.dir / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    for metric in METRICS:
        r = result[metric]
        print(
            f"{metric:9s} CodeBERT {r['codebert']:.3f}  TF-IDF {r['tfidf']:.3f}  "
            f"only CodeBERT {r['b']}  only TF-IDF {r['c']}  exact p = {r['exact_p']:.2g}"
        )


if __name__ == "__main__":
    main()
