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
