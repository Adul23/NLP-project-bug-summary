"""New baseline protocol; preserve the historical baseline and its artifacts."""

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np

from .experiment_data import DEFAULT_SPLITS, load_splits
from .split_dataset import ROOT
from .train_tfidf import (
    error_examples,
    evaluate_predictions,
    mutation_results,
    save_models,
    select_model,
)


def train(split_dir, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Use an empty output directory to preserve completed experiments"
        )
    (training, validation, test), manifest = load_splits(split_dir)
    vectorizer, model, comparison = select_model(
        training, validation, sublinear_tf=False
    )
    selected = next(
        row for row in comparison if tuple(row["ngram_range"]) == vectorizer.ngram_range
    )
    features = vectorizer.transform(test.code)
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, list(model.classes_).index(1)]
    report = {
        "model": "TF-IDF + LR",
        "split_manifest": manifest,
        "status": "trained",
        "config": {
            "ngram_range": vectorizer.ngram_range,
            "sublinear_tf": False,
            "max_features": 30000,
            "lowercase": False,
            "seed": 42,
        },
        "versions": {
            **{
                name: version(name)
                for name in ("scikit-learn", "numpy", "pandas", "joblib")
            },
            "python": platform.python_version(),
        },
        "logistic_regression": {
            "max_iter": 2000,
            "class_weight": "balanced",
            "random_state": 42,
        },
        "comparison": comparison,
        "validation": selected["validation"],
        "test": evaluate_predictions(test.label, predictions),
        "mutation_types": mutation_results(test, predictions),
        "error_examples": error_examples(test, predictions, probabilities),
    }
    save_models(vectorizer, model, output)
    loaded_v = joblib.load(output / "tfidf_vectorizer.joblib")
    loaded_m = joblib.load(output / "logistic_regression.joblib")
    # Check reload on validation; do not re-evaluate the held-out test set.
    probe = validation.code.iloc[:32]
    np.testing.assert_array_equal(
        model.predict(vectorizer.transform(probe)),
        loaded_m.predict(loaded_v.transform(probe)),
    )
    np.testing.assert_allclose(
        model.predict_proba(vectorizer.transform(probe)),
        loaded_m.predict_proba(loaded_v.transform(probe)),
    )
    report["reload_predictions_identical"] = True
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["test"], indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output", type=Path, default=ROOT / "models/tfidf_v2")
    args = parser.parse_args()
    train(args.split_dir, args.output)


if __name__ == "__main__":
    main()
