import random

import pandas as pd
from src.data.mutations import mutate_code
from src.evaluation.localization_eval import evaluate


class OraclePredictor:
    """Low bug probability exactly for the known original functions."""

    def __init__(self, originals):
        self.originals = set(originals)

    def predict_batch(self, codes):
        return [{"bug_probability": 0.1 if c in self.originals else 0.9} for c in codes]


def make_test_split():
    rng = random.Random(0)
    originals = [
        "def f(a, b):\n    if a > b:\n        return a - b\n    return b\n",
        "def g(xs):\n    return [x * 2 for x in xs if x and x != 0]\n",
        "def h(n):\n    return n + 1\n",
    ]
    rows = []
    for i, original in enumerate(originals):
        mutant = mutate_code(original, rng)
        rows.append(
            {
                "sample_id": 2 * i,
                "source_id": f"{i:06d}",
                "code": original,
                "label": 0,
                "mutation_type": "clean",
            }
        )
        rows.append(
            {
                "sample_id": 2 * i + 1,
                "source_id": f"{i:06d}",
                "code": mutant.code,
                "label": 1,
                "mutation_type": mutant.mutation_type,
            }
        )
    rows.append(
        {
            "sample_id": 99,
            "source_id": "999999",
            "code": "def z(x):\n    return x\n",
            "label": 0,
            "mutation_type": "clean",
        }
    )
    return pd.DataFrame(rows), originals


def test_oracle_gets_perfect_top1():
    test, originals = make_test_split()
    rows, summary = evaluate(test, OraclePredictor(originals), progress=False)
    assert len(rows) == 3  # one per buggy function; the clean-only source is skipped
    assert summary["all"]["top1"] == 1.0 and summary["all"]["top3"] == 1.0
    assert summary["all"]["line_top1"] == 1.0
    assert summary["all"]["true_fix_missing"] == 0
    assert 0 < summary["all"]["random_top1"] <= 1
    assert (
        summary["classifier_detected"]["n"] == 3
        and summary["classifier_missed"] is None
    )


def test_constant_predictor_is_not_better_than_ranking_order():
    test, _ = make_test_split()

    class Constant:
        def predict_batch(self, codes):
            return [{"bug_probability": 0.5} for _ in codes]

    rows, summary = evaluate(test, Constant(), progress=False)
    assert summary["all"]["top3"] >= summary["all"]["top1"]
    assert summary["classifier_detected"]["n"] == 3  # 0.5 >= default threshold 0.5
