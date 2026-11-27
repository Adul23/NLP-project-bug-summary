import random

from src.bug_detection.localize import REPAIRS, repair_candidates
from src.data.mutations import mutate_code


def test_repairs_invert_mutation_tables():
    assert sorted(REPAIRS[">"]) == ["<", ">="]
    assert REPAIRS["+"] == ["-"]
    assert ">=" not in REPAIRS  # nothing mutates into >=


def test_true_original_is_always_a_candidate():
    rng = random.Random(0)
    originals = [
        "def f(a, b):\n    return a >= b",
        "def g(x):\n    return x * 2 - 1",
        "def h(a, b):\n    # a > b in comment\n    return a or b",
        "def k(xs, n):\n    return [x for x in xs if x <= n and x == 1]",
    ]
    for original in originals:
        for _ in range(5):
            mutant = mutate_code(original, rng)
            assert original in [c["code"] for c in repair_candidates(mutant.code)]


def test_candidate_fields():
    code = "def f(x):\n    if x < 0:\n        return x + 1\n    return x\n"
    found = {(c["line"], c["old"], c["new"]) for c in repair_candidates(code)}
    assert found == {(2, "<", ">"), (2, "<", "<="), (3, "+", "-")}


class FakePredictor:
    """Low bug probability only for the one known-correct version of the code."""

    def __init__(self, correct):
        self.correct = correct
        self.calls = []

    def predict_batch(self, codes):
        self.calls.append(len(codes))
        return [{"bug_probability": 0.1 if c == self.correct else 0.9} for c in codes]


def test_localize_ranks_true_fix_first():
    from src.bug_detection.localize import localize

    original = "def f(xs, n):\n    total = 0\n    for x in xs:\n        if x > n:\n            total = total + x\n    return total\n"
    buggy = original.replace("x > n", "x < n")
    predictor = FakePredictor(original)
    result = localize(buggy, predictor)
    best = result["candidates"][0]
    assert best["code"] == original
    assert (best["line"], best["old"], best["new"]) == (4, "<", ">")
    assert abs(best["drop"] - 0.8) < 1e-9
    assert predictor.calls == [1 + result["n_candidates"]]  # one batch for everything
    assert len(result["candidates"]) <= 3


def test_localize_without_operators():
    from src.bug_detection.localize import localize

    result = localize("def f(x):\n    return x\n", FakePredictor(None))
    assert result["n_candidates"] == 0 and result["candidates"] == []
