"""Exercise training, held-out isolation, evaluation and artifact inference."""

import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def fixture_frame(prefix, count=20):
    return pd.DataFrame(
        [
            {
                "source_id": f"{prefix}{i:06d}",
                "code": f"def f{i}(x): return x {'+' if label == 0 else '-'} 1",
                "label": label,
                "mutation_type": "clean" if label == 0 else "arithmetic_operator",
            }
            for i in range(count)
            for label in (0, 1)
        ]
    )


def test_training_excludes_validation_text_and_saves_predictable_models(
    tmp_path, monkeypatch
):
    from src.bug_detection import predict_tfidf
    from src.bug_detection.train_tfidf import save_models, select_model

    train = fixture_frame("train")
    validation = fixture_frame("validation")
    validation["code"] += " # ZZZZZ"
    vectorizer, model, comparison = select_model(train, validation)
    assert "ZZZ" not in vectorizer.vocabulary_
    assert len(comparison) == 3
    assert {tuple(row["ngram_range"]) for row in comparison} == {(2, 4), (3, 5), (3, 6)}
    best = next(
        row for row in comparison if tuple(row["ngram_range"]) == vectorizer.ngram_range
    )
    assert best["f1"] == max(row["f1"] for row in comparison)
    predictions = model.predict(vectorizer.transform(train.code))
    assert predictions.tolist() == train.label.tolist()
    assert np.all(
        (model.predict_proba(vectorizer.transform(validation.code)) >= 0)
        & (model.predict_proba(vectorizer.transform(validation.code)) <= 1)
    )
    save_models(vectorizer, model, tmp_path)
    monkeypatch.setattr(predict_tfidf, "MODEL_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    for code, label in [
        ("def f(x): return x + 1", "clean"),
        ("def f(x): return x - 1", "buggy"),
    ]:
        result = predict_tfidf.predict_bug(code)
        assert set(result) == {"label", "bug_probability"}
        assert result["label"] == label
        assert 0 <= result["bug_probability"] <= 1
    loaded = joblib.load(tmp_path / "tfidf_vectorizer.joblib")
    assert "ZZZ" not in loaded.vocabulary_


@pytest.mark.parametrize(
    "candidate_predictions,winner",
    [
        ([[0, 0, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0]], (2, 4)),
        ([[0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1]], (3, 5)),
    ],
)
def test_selection_uses_highest_validation_f1_and_first_on_ties(
    monkeypatch, candidate_predictions, winner
):
    from src.bug_detection import train_tfidf

    real_classifier = train_tfidf.LogisticRegression
    next_candidate = iter(candidate_predictions)

    class ControlledPredictions(real_classifier):
        # Real fitting and features; control only classifier outputs so selection
        # has known unequal scores independent of solver/library changes.
        def predict(self, features):
            return np.array(next(next_candidate))

    monkeypatch.setattr(train_tfidf, "LogisticRegression", ControlledPredictions)
    vectorizer, _, _ = train_tfidf.select_model(
        fixture_frame("train"), fixture_frame("val", count=2)
    )
    assert vectorizer.ngram_range == winner


def test_full_run_fits_only_train_and_evaluates_test_once(tmp_path, monkeypatch):
    from src.bug_detection import train_tfidf
    from src.bug_detection.split_dataset import split_dataset

    frame = fixture_frame("", count=40)
    train, validation, test = split_dataset(frame)
    input_path = tmp_path / "dataset.csv"
    frame.to_csv(input_path, index=False)
    events = []
    original_fit_transform = train_tfidf.TfidfVectorizer.fit_transform
    original_fit = train_tfidf.LogisticRegression.fit
    original_transform = train_tfidf.TfidfVectorizer.transform
    original_evaluate = train_tfidf.evaluate_predictions

    def fit_transform(self, documents, *args, **kwargs):
        events.append(("fit_tfidf", tuple(documents)))
        return original_fit_transform(self, documents, *args, **kwargs)

    def fit(self, features, labels, *args, **kwargs):
        events.append(("fit_model", len(labels)))
        return original_fit(self, features, labels, *args, **kwargs)

    def transform(self, documents):
        events.append(("transform", tuple(documents)))
        return original_transform(self, documents)

    def evaluate(labels, predictions):
        events.append(("evaluate", len(labels)))
        return original_evaluate(labels, predictions)

    monkeypatch.setattr(train_tfidf.TfidfVectorizer, "fit_transform", fit_transform)
    monkeypatch.setattr(train_tfidf.TfidfVectorizer, "transform", transform)
    monkeypatch.setattr(train_tfidf.LogisticRegression, "fit", fit)
    monkeypatch.setattr(train_tfidf, "evaluate_predictions", evaluate)
    train_tfidf.run_baseline(
        input_path, tmp_path / "splits", tmp_path / "models", tmp_path / "results"
    )
    expected = []
    for _ in range(3):
        expected.extend(
            [
                ("fit_tfidf", tuple(train.code)),
                ("transform", tuple(validation.code)),
                ("fit_model", len(train)),
                ("evaluate", len(validation)),
            ]
        )
    expected.extend([("transform", tuple(test.code)), ("evaluate", len(test))])
    assert events == expected


def test_metrics_and_mutation_recall_use_buggy_samples_only():
    from src.bug_detection.train_tfidf import (
        error_examples,
        evaluate_predictions,
        mutation_results,
    )

    frame = pd.DataFrame(
        {
            "source_id": ["000001", "000002", "000003", "000004"],
            "code": ["code a", "code b", "code c", "code d"],
            "label": [0, 0, 1, 1],
            "mutation_type": [
                "clean",
                "clean",
                "comparison_operator",
                "comparison_operator",
            ],
        }
    )
    predicted = np.array([0, 1, 0, 1])
    probabilities = np.array([0.1, 0.8, 0.4, 0.9])
    metrics = evaluate_predictions(frame.label, predicted)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert (
        metrics["accuracy"]
        == metrics["bug_precision"]
        == metrics["bug_recall"]
        == metrics["bug_f1"]
        == 0.5
    )
    assert metrics["classification_report"]["clean"]["support"] == 2
    assert metrics["classification_report"]["buggy"]["support"] == 2
    mutations = mutation_results(frame, predicted)
    assert mutations["comparison_operator"] == {
        "samples": 2,
        "correctly_detected": 1,
        "recall": 0.5,
    }
    assert mutations["arithmetic_operator"] == {
        "samples": 0,
        "correctly_detected": 0,
        "recall": None,
    }
    assert mutations["boolean_operator"]["samples"] == 0
    examples = error_examples(frame, predicted, probabilities)
    assert [
        examples[name][0]["source_id"]
        for name in (
            "true_positives",
            "true_negatives",
            "false_positives",
            "false_negatives",
        )
    ] == ["000004", "000001", "000002", "000003"]
    assert examples["false_negatives"][0]["bug_probability"] == 0.4
    assert examples["false_positives"][0]["true_label"] == "CLEAN"


def test_examples_limit_and_no_predicted_bugs():
    from src.bug_detection.train_tfidf import error_examples, evaluate_predictions

    frame = fixture_frame("example")
    predicted = np.zeros(len(frame), dtype=int)
    metrics = evaluate_predictions(frame.label, predicted)
    assert metrics["bug_precision"] == metrics["bug_recall"] == metrics["bug_f1"] == 0
    examples = error_examples(frame, predicted, np.full(len(frame), 0.2))
    assert len(examples["false_negatives"]) == 5
    assert len(examples["true_negatives"]) == 5
    assert examples["true_positives"] == examples["false_positives"] == []


@pytest.mark.parametrize("code", [None, 123, "", "   "])
def test_predict_rejects_invalid_code_before_loading(code):
    from src.bug_detection.predict_tfidf import predict_bug

    with pytest.raises((TypeError, ValueError)):
        predict_bug(code)


def test_missing_models_explain_training_command(tmp_path, monkeypatch):
    from src.bug_detection import predict_tfidf

    monkeypatch.setattr(predict_tfidf, "MODEL_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="train_tfidf"):
        predict_tfidf.predict_bug("def f(): return 1")


def test_cli_produces_splits_reports_and_models(tmp_path):
    dataset = tmp_path / "dataset.csv"
    fixture_frame("", count=40).to_csv(dataset, index=False)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "src/bug_detection/train_tfidf.py"),
            "--input",
            str(dataset),
            "--split-dir",
            str(tmp_path / "splits"),
            "--model-dir",
            str(tmp_path / "models"),
            "--results-dir",
            str(tmp_path / "results"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(
        (tmp_path / "results/tfidf_logistic_regression_metrics.json").read_text()
    )
    assert report["dataset"]["samples"] == 80
    assert report["source_id_overlap"] == {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
    }
    assert len(report["configuration_comparison"]) == 3
    assert report["test"]["accuracy"] == report["accuracy"]
    assert (
        sum(map(sum, report["confusion_matrix"])) == report["splits"]["test"]["samples"]
    )
    assert report["validation"]["bug_f1"] == max(
        row["f1"] for row in report["configuration_comparison"]
    )
    assert (tmp_path / "models/logistic_regression.joblib").exists()
    assert (tmp_path / "models/tfidf_vectorizer.joblib").exists()
    assert (
        "FALSE NEGATIVES" in (tmp_path / "results/tfidf_error_examples.txt").read_text()
    )
    for name in ("train", "validation", "test"):
        assert (tmp_path / f"splits/{name}.csv").exists()
