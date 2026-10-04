from __future__ import annotations

import pandas as pd
import pytest

from src.summarization import bm25_retrieval
from src.summarization.evaluate_baseline import evaluate


def test_retrieves_summary_of_best_matching_train_function():
    train_codes = [
        "def parse_json_config(path): return load_json(path)",
        "def draw_circle(canvas, radius): canvas.circle(radius)",
    ]
    train_summaries = ["Parse a JSON configuration file.", "Draw a circle on the canvas."]
    preds, no_overlap = bm25_retrieval.retrieve_summaries(
        train_codes, train_summaries, ["def load_json_settings(p): return parse_json(p)"]
    )
    assert preds == ["Parse a JSON configuration file."]
    assert no_overlap == 0


def test_rare_shared_token_beats_common_token():
    # "common" appears in every training function; "zebra" appears only in the second one.
    # A query that shares only "zebra" with doc 2 must pick doc 2 over the common-token matches.
    train_codes = [
        "common alpha beta",
        "common gamma delta",
        "common zebra",
    ]
    train_summaries = ["first", "second", "third"]
    preds, _ = bm25_retrieval.retrieve_summaries(
        train_codes, train_summaries, ["zebra common"]
    )
    assert preds == ["third"]


def test_counts_queries_without_token_overlap():
    preds, no_overlap = bm25_retrieval.retrieve_summaries(
        ["p = alpha"], ["Alpha."], ["q = zzzz"]
    )
    assert preds == ["Alpha."]
    assert no_overlap == 1


def test_index_uses_only_train_rows():
    # Only the train row is passed as the index. A validation copy of the query never enters it.
    train = pd.DataFrame(
        [{"code": "def draw_circle(r): pass", "summary": "Draw a circle.", "split": "train"}]
    )
    preds, _ = bm25_retrieval.retrieve_summaries(
        train["code"].tolist(), train["summary"].tolist(),
        ["def compute_total(items): return sum(items)"],
    )
    assert preds == ["Draw a circle."]


def test_empty_index_raises():
    with pytest.raises(ValueError):
        bm25_retrieval.retrieve_summaries([], [], ["def f(): pass"])


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        bm25_retrieval.retrieve_summaries(["def f(): pass"], [], ["def g(): pass"])


def test_evaluate_bm25_on_small_csv(tmp_path):
    dataset = tmp_path / "summ.csv"
    pd.DataFrame(
        [
            {"source_id": "000001", "code": "def parse_json_config(p): return load(p)", "summary": "Parse the JSON config file.", "split": "train"},
            {"source_id": "000002", "code": "def draw_circle(r): pass", "summary": "Draw a circle on the canvas.", "split": "train"},
            {"source_id": "000003", "code": "def parse_json_settings(p): return load(p)", "summary": "Parse the JSON config file.", "split": "validation"},
        ]
    ).to_csv(dataset, index=False)

    result = evaluate(dataset, "validation", method="bm25")

    assert result["method"] == "bm25"
    assert result["examples"] == 1
    assert result["bleu4"] == pytest.approx(1.0)
    assert result["rouge_l_f1"] == pytest.approx(1.0)
