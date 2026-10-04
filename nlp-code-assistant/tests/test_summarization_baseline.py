from __future__ import annotations

import pandas as pd
import pytest

from src.summarization import ast_baseline
from src.summarization.evaluate_baseline import evaluate
from src.summarization.metrics import corpus_bleu, mean_rouge_l, rouge_l_f1


def test_bleu_identical_is_one():
    assert corpus_bleu(["return the sum of values"], ["return the sum of values"]) == pytest.approx(1.0)


def test_bleu_disjoint_is_zero():
    assert corpus_bleu(["alpha beta gamma delta"], ["one two three four"]) == 0.0


def test_bleu_short_hypothesis_is_penalized():
    ref = ["read the config file from disk now"]
    hyp = ["read the config file"]
    score = corpus_bleu(ref, hyp)
    assert 0.0 < score < 1.0


def test_bleu_length_mismatch_raises():
    with pytest.raises(ValueError):
        corpus_bleu(["a"], [])


def test_rouge_l_hand_computed():
    # LCS("return sum", "return the sum") = 2; P = 2/3, R = 2/2 -> F1 = 0.8
    assert rouge_l_f1("return sum", "return the sum") == pytest.approx(0.8)


def test_rouge_l_ignores_case_and_punctuation():
    assert rouge_l_f1("Get user name.", "get user name") == pytest.approx(1.0)


def test_rouge_l_empty_is_zero():
    assert rouge_l_f1("", "anything") == 0.0
    assert rouge_l_f1("anything", "") == 0.0


def test_mean_rouge_l_empty_raises():
    with pytest.raises(ValueError):
        mean_rouge_l([], [])


@pytest.mark.parametrize(
    "code, expected",
    [
        ("def get_user_name(self):\n    return self.name", "Get user name."),
        ("def total_value(x):\n    return x * 2", "Return the total value."),
        ("def total_value(x):\n    print(x)", "Total value."),
        ("def __init__(self):\n    pass", "Initialize the object."),
        ("async def load_file(path):\n    return path", "Load file."),
        ("def broken(:", "Perform the operation."),
        ("x = 1", "Perform the operation."),
    ],
)
def test_ast_baseline_templates(code, expected):
    assert ast_baseline.predict(code) == expected


def test_ast_baseline_only_uses_first_function():
    code = "def add_pair(a, b):\n    return a + b\n\ndef other_thing():\n    pass"
    assert ast_baseline.predict(code) == "Add pair."


def test_evaluate_on_small_csv(tmp_path):
    dataset = tmp_path / "summ.csv"
    pd.DataFrame(
        [
            {"source_id": "000001", "code": "def get_user_name(self):\n    return 1", "summary": "Get user name.", "split": "validation"},
            {"source_id": "000002", "code": "def total_value(x):\n    return x", "summary": "Return the total value.", "split": "validation"},
            {"source_id": "000003", "code": "def other(x):\n    return x", "summary": "Something else.", "split": "train"},
        ]
    ).to_csv(dataset, index=False)

    result = evaluate(dataset, "validation")

    assert result["split"] == "validation"
    assert result["examples"] == 2
    assert 0.0 <= result["bleu4"] <= 1.0
    assert 0.0 <= result["rouge_l_f1"] <= 1.0


def test_evaluate_rejects_missing_split(tmp_path):
    dataset = tmp_path / "summ.csv"
    pd.DataFrame(
        [{"source_id": "000001", "code": "def f():\n    return 1", "summary": "F.", "split": "train"}]
    ).to_csv(dataset, index=False)
    with pytest.raises(ValueError):
        evaluate(dataset, "test")
