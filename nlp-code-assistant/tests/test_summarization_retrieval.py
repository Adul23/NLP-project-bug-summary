from __future__ import annotations

import pandas as pd
import pytest

from src.summarization import tfidf_retrieval
from src.summarization.evaluate_baseline import evaluate


def test_code_tokens_split_snake_and_camel_case():
    assert tfidf_retrieval.code_tokens("def getUserName_v2(x): pass") == [
        "def", "get", "user", "name", "v2", "x", "pass",
    ]


def test_retrieves_summary_of_most_similar_train_function():
    train_codes = [
        "def parse_json_config(path): return load_json(path)",
        "def draw_circle(canvas, radius): canvas.circle(radius)",
    ]
    train_summaries = ["Parse a JSON configuration file.", "Draw a circle on the canvas."]
    preds, no_overlap = tfidf_retrieval.retrieve_summaries(
        train_codes, train_summaries, ["def load_json_settings(p): return parse_json(p)"]
    )
    assert preds == ["Parse a JSON configuration file."]
    assert no_overlap == 0


def test_counts_queries_without_token_overlap():
    preds, no_overlap = tfidf_retrieval.retrieve_summaries(
        ["p = alpha"], ["Alpha."], ["q = zzzz"]
    )
    assert preds == ["Alpha."]
    assert no_overlap == 1


def test_index_uses_only_train_rows():
    # The validation row is an exact copy of the query, but it must not be a candidate.
    train = pd.DataFrame(
        [{"code": "def draw_circle(r): pass", "summary": "Draw a circle.", "split": "train"}]
    )
    validation = pd.DataFrame(
        [
            {"code": "def compute_total(items): return sum(items)", "summary": "Wrong.", "split": "validation"},
        ]
    )
    query = validation.copy()
    preds, _ = tfidf_retrieval.predict_split(train, query)
    assert preds == ["Draw a circle."]


def test_empty_index_raises():
    with pytest.raises(ValueError):
        tfidf_retrieval.retrieve_summaries([], [], ["def f(): pass"])


def test_evaluate_tfidf_on_small_csv(tmp_path):
    dataset = tmp_path / "summ.csv"
    pd.DataFrame(
        [
            {"source_id": "000001", "code": "def parse_json_config(p): return load(p)", "summary": "Parse the JSON config file.", "split": "train"},
            {"source_id": "000002", "code": "def draw_circle(r): pass", "summary": "Draw a circle on the canvas.", "split": "train"},
            {"source_id": "000003", "code": "def parse_json_settings(p): return load(p)", "summary": "Parse the JSON config file.", "split": "validation"},
        ]
    ).to_csv(dataset, index=False)

    result = evaluate(dataset, "validation", method="tfidf")

    assert result["method"] == "tfidf"
    assert result["examples"] == 1
    assert result["bleu4"] == pytest.approx(1.0)
    assert result["rouge_l_f1"] == pytest.approx(1.0)


def test_evaluate_rejects_unknown_method(tmp_path):
    dataset = tmp_path / "summ.csv"
    pd.DataFrame(
        [{"source_id": "000001", "code": "def f(): pass", "summary": "F.", "split": "validation"}]
    ).to_csv(dataset, index=False)
    with pytest.raises(ValueError):
        evaluate(dataset, "validation", method="bm25")


def test_evaluate_tfidf_requires_train_rows(tmp_path):
    dataset = tmp_path / "summ.csv"
    pd.DataFrame(
        [{"source_id": "000001", "code": "def f(): pass", "summary": "F.", "split": "validation"}]
    ).to_csv(dataset, index=False)
    with pytest.raises(ValueError):
        evaluate(dataset, "validation", method="tfidf")
