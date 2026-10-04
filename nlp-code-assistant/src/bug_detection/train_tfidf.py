"""Train-only TF-IDF fitting, validation selection, and one final test evaluation."""

import argparse
import hashlib
import json
import platform
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

if __package__:
    from .split_dataset import (
        ROOT,
        SPLIT_NAMES,
        dataset_statistics,
        read_dataset,
        save_splits,
        split_dataset,
        validate_frame,
        validate_splits,
    )
else:
    from split_dataset import (
        ROOT,
        SPLIT_NAMES,
        dataset_statistics,
        read_dataset,
        save_splits,
        split_dataset,
        validate_frame,
        validate_splits,
    )

NGRAM_RANGES = ((3, 5), (2, 4), (3, 6))
MUTATION_TYPES = ("comparison_operator", "arithmetic_operator", "boolean_operator")


def evaluate_predictions(labels, predictions) -> dict:
    report = classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=["clean", "buggy"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(np.mean(np.asarray(labels) == np.asarray(predictions))),
        "bug_precision": report["buggy"]["precision"],
        "bug_recall": report["buggy"]["recall"],
        "bug_f1": report["buggy"]["f1-score"],
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
    }


def select_model(train: pd.DataFrame, validation: pd.DataFrame) -> tuple:
    """Select on validation Buggy F1; the test set is not an input to this step."""
    for frame in (train, validation):
        validate_frame(frame)
        if set(frame["label"]) != {0, 1}:
            raise ValueError("Training and validation require both classes")
    if set(train["source_id"]) & set(validation["source_id"]):
        raise ValueError("Training/validation source_id overlap")
    best_score = -1.0
    best_vectorizer = best_model = None
    comparison = []
    for ngram_range in NGRAM_RANGES:
        print(f"Training character TF-IDF {ngram_range} ...", flush=True)
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            max_features=30000,
            lowercase=False,
            sublinear_tf=True,
        )
        X_train = vectorizer.fit_transform(train["code"])
        X_validation = vectorizer.transform(validation["code"])
        model = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(X_train, train["label"])
        metrics = evaluate_predictions(validation["label"], model.predict(X_validation))
        comparison.append(
            {
                "ngram_range": list(ngram_range),
                "precision": metrics["bug_precision"],
                "recall": metrics["bug_recall"],
                "f1": metrics["bug_f1"],
                "validation": metrics,
            }
        )
        print(f"Validation Buggy F1: {metrics['bug_f1']:.6f}", flush=True)
        if metrics["bug_f1"] > best_score:
            best_score = metrics["bug_f1"]
            best_vectorizer, best_model = vectorizer, model
    return (
        best_vectorizer,
        best_model,
        sorted(comparison, key=lambda row: row["ngram_range"]),
    )


def mutation_results(frame: pd.DataFrame, predictions) -> dict:
    predictions = np.asarray(predictions)
    buggy = frame["label"].to_numpy() == 1
    categories = set(MUTATION_TYPES) | set(frame.loc[buggy, "mutation_type"])
    results = {}
    for category in sorted(categories):
        mask = buggy & (frame["mutation_type"].to_numpy() == category)
        samples = int(mask.sum())
        detected = int((predictions[mask] == 1).sum())
        results[category] = {
            "samples": samples,
            "correctly_detected": detected,
            "recall": detected / samples if samples else None,
        }
    return results


def error_examples(
    frame: pd.DataFrame, predictions, probabilities, limit: int = 5
) -> dict:
    labels = frame["label"].to_numpy()
    predictions, probabilities = np.asarray(predictions), np.asarray(probabilities)
    outcomes = {
        "true_positives": (1, 1),
        "true_negatives": (0, 0),
        "false_positives": (0, 1),
        "false_negatives": (1, 0),
    }
    examples = {}
    for name, (truth, predicted) in outcomes.items():
        positions = np.flatnonzero((labels == truth) & (predictions == predicted))[
            :limit
        ]
        examples[name] = [
            {
                "source_id": str(frame.iloc[index]["source_id"]),
                "true_label": "BUGGY" if truth else "CLEAN",
                "predicted_label": "BUGGY" if predicted else "CLEAN",
                "bug_probability": float(probabilities[index]),
                "mutation_type": frame.iloc[index]["mutation_type"],
                "code": frame.iloc[index]["code"],
            }
            for index in positions
        ]
    return examples


def format_examples(examples: dict) -> str:
    sections = []
    for name, records in examples.items():
        sections.append(f"\n{name.replace('_', ' ').upper()} ({len(records)} examples)")
        for record in records:
            sections.append("-" * 40)
            sections.extend(
                f"{key}: {record[key]}"
                for key in (
                    "source_id",
                    "true_label",
                    "predicted_label",
                    "bug_probability",
                    "mutation_type",
                )
            )
            sections.append(f"\nCODE:\n{record['code']}")
    return "\n".join(sections) + "\n"


def save_models(vectorizer, model, model_dir: Path) -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, model_dir / "tfidf_vectorizer.joblib")
    joblib.dump(model, model_dir / "logistic_regression.joblib")


def print_metrics(title: str, metrics: dict) -> None:
    print(f"\n{title}\n              precision    recall        f1   support")
    for label in ("clean", "buggy"):
        row = metrics["classification_report"][label]
        print(
            f"{label:12} {row['precision']:10.4f} {row['recall']:9.4f} {row['f1-score']:9.4f} {int(row['support']):9}"
        )
    print(f"Accuracy: {metrics['accuracy']:.4f}\nBug F1: {metrics['bug_f1']:.4f}")
    print("Confusion Matrix [[TN FP] [FN TP]]:")
    print(np.asarray(metrics["confusion_matrix"]))


def run_baseline(
    input_path: Path, split_dir: Path, model_dir: Path, results_dir: Path
) -> dict:
    frame = read_dataset(input_path)
    train, validation, test = splits = split_dataset(frame)
    overlap = validate_splits(*splits)
    save_splits(splits, split_dir)
    statistics = dataset_statistics(frame)
    split_stats = {
        name: dataset_statistics(part) for name, part in zip(SPLIT_NAMES, splits)
    }
    print("DATASET", json.dumps(statistics), flush=True)
    print("SPLITS", json.dumps(split_stats), flush=True)
    print("SOURCE_ID OVERLAP", json.dumps(overlap), flush=True)

    vectorizer, model, comparison = select_model(train, validation)
    selected = next(
        row for row in comparison if row["ngram_range"] == list(vectorizer.ngram_range)
    )
    print("\nngram_range | precision | recall | f1")
    for row in comparison:
        print(
            f"{tuple(row['ngram_range'])} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f}"
        )
    print(f"Selected ngram_range: {vectorizer.ngram_range}", flush=True)
    print_metrics("VALIDATION RESULTS", selected["validation"])

    # The trained winner is retained. No refit, threshold tuning, or test-based selection.
    X_test = vectorizer.transform(test["code"])
    predictions = model.predict(X_test)
    bug_column = int(np.flatnonzero(model.classes_ == 1)[0])
    probabilities = model.predict_proba(X_test)[:, bug_column]
    test_metrics = evaluate_predictions(test["label"], predictions)
    by_mutation = mutation_results(test, predictions)
    examples = error_examples(test, predictions, probabilities)
    report = {
        "model": "TF-IDF + Logistic Regression",
        "ngram_range": list(vectorizer.ngram_range),
        **test_metrics,
        "validation": selected["validation"],
        "test": test_metrics,
        "dataset": statistics,
        "splits": split_stats,
        "source_id_overlap": overlap,
        "configuration_comparison": comparison,
        "mutation_types": by_mutation,
        "error_examples": examples,
        "protocol": {
            "seed": 42,
            "source_id_split_ratios": [0.70, 0.15, 0.15],
            "selection_metric": "validation buggy F1",
            "fit_partition": "train",
            "tie_break": "first evaluated; order (3,5), (2,4), (3,6)",
            "test_evaluations": 1,
            "refit_on_validation": False,
        },
        "tfidf": {
            "analyzer": "char",
            "max_features": 30000,
            "lowercase": False,
            "sublinear_tf": True,
            "vocabulary_size": len(vectorizer.vocabulary_),
        },
        "logistic_regression": {
            "max_iter": 2000,
            "class_weight": "balanced",
            "random_state": 42,
        },
        "dataset_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    save_models(vectorizer, model, model_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "tfidf_logistic_regression_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    examples_text = format_examples(examples)
    (results_dir / "tfidf_error_examples.txt").write_text(
        examples_text, encoding="utf-8"
    )
    print_metrics("TEST RESULTS", test_metrics)
    print("\nRESULTS BY MUTATION TYPE\n" + json.dumps(by_mutation, indent=2))
    print(examples_text)
    print(f"Models saved to {model_dir}\nResults saved to {results_dir}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "data/processed/bug_detection_dataset.csv"
    )
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    run_baseline(args.input, args.split_dir, args.model_dir, args.results_dir)


if __name__ == "__main__":
    main()
