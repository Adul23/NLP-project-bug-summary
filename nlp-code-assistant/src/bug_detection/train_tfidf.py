"""Train character TF-IDF + Logistic Regression; select on validation only."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

if __package__:
    from .split_dataset import (DATASET_PATH, MUTATION_TYPES, PROJECT_ROOT, SPLIT_NAMES,
                                check_source_overlap, dataset_statistics, load_dataset,
                                print_summary, save_splits, split_dataset)
else:
    from split_dataset import (DATASET_PATH, MUTATION_TYPES, PROJECT_ROOT, SPLIT_NAMES,
                               check_source_overlap, dataset_statistics, load_dataset,
                               print_summary, save_splits, split_dataset)

NGRAM_RANGES = ((2, 4), (3, 5), (3, 6))


def fit_candidate(train, ngram_range=(3, 5)):
    if set(train['label']) != {0, 1}:
        raise ValueError('Training label must contain both CLEAN (0) and BUGGY (1)')
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=ngram_range,
                                 max_features=30000, lowercase=False, sublinear_tf=True)
    matrix = vectorizer.fit_transform(train['code'])
    model = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
    # Do not silently publish a model that failed to converge.
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(matrix, train['label'])
    return vectorizer, model


def evaluate(vectorizer, model, frame):
    matrix = vectorizer.transform(frame['code'])
    predicted = model.predict(matrix)
    bug_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(matrix)[:, bug_index]
    report = classification_report(frame['label'], predicted, labels=[0, 1],
                                   target_names=['clean', 'buggy'], output_dict=True,
                                   zero_division=0)
    metrics = {
        'accuracy': float(accuracy_score(frame['label'], predicted)),
        'bug_precision': report['buggy']['precision'],
        'bug_recall': report['buggy']['recall'], 'bug_f1': report['buggy']['f1-score'],
        'classification_report': report,
        'confusion_matrix': confusion_matrix(frame['label'], predicted, labels=[0, 1]).tolist(),
    }
    return metrics, predicted, probabilities


def print_metrics(title, metrics):
    print(f'\n{title}\n')
    print(f'{"class":12} {"precision":>10} {"recall":>10} {"f1-score":>10} {"support":>10}')
    for name in ('clean', 'buggy'):
        row = metrics['classification_report'][name]
        print(f'{name:12} {row["precision"]:10.4f} {row["recall"]:10.4f} '
              f'{row["f1-score"]:10.4f} {int(row["support"]):10}')
    print(f'Accuracy: {metrics["accuracy"]:.4f}')
    print(f'Buggy Precision: {metrics["bug_precision"]:.4f}')
    print(f'Buggy Recall: {metrics["bug_recall"]:.4f}')
    print(f'Buggy F1: {metrics["bug_f1"]:.4f}')
    print('Confusion Matrix [[TN FP], [FN TP]]:')
    print(np.asarray(metrics['confusion_matrix']))


def mutation_analysis(frame, predicted):
    kinds = dict.fromkeys((*MUTATION_TYPES, *sorted(frame.loc[frame.label.eq(1), 'mutation_type'].unique())))
    result = {}
    for kind in kinds:
        mask = (frame.label.eq(1) & frame.mutation_type.eq(kind)).to_numpy()
        count = int(mask.sum())
        detected = int((predicted[mask] == 1).sum())
        result[kind] = {'samples': count, 'correctly_detected': detected,
                        'recall': detected / count if count else None}
    return result


def error_examples(frame, predicted, probabilities, limit=5):
    """Use shortest examples for readable reports; retain full code and probabilities."""
    rows = frame.copy()
    rows['predicted_label'] = predicted
    rows['bug_probability'] = probabilities
    rows['length'] = rows.code.str.len()
    groups = {}
    for name, true, pred in [('true_positives', 1, 1), ('true_negatives', 0, 0),
                             ('false_positives', 0, 1), ('false_negatives', 1, 0)]:
        selected = rows[rows.label.eq(true) & rows.predicted_label.eq(pred)]
        selected = selected.sort_values('length', kind='stable').head(limit)
        groups[name] = [
            {'source_id': row.source_id, 'true_label': 'buggy' if true else 'clean',
             'predicted_label': 'buggy' if pred else 'clean',
             'bug_probability': float(row.bug_probability),
             'mutation_type': row.mutation_type, 'code': row.code}
            for row in selected.itertuples(index=False)
        ]
    return groups


def format_examples(groups):
    output = []
    for name, examples in groups.items():
        output.append(f'\n{name.upper()}: {len(examples)} examples (up to 5 available)')
        for row in examples:
            output.append('-' * 40)
            output.extend(f'{key}: {row[key]}' for key in
                          ('source_id', 'true_label', 'predicted_label', 'bug_probability', 'mutation_type'))
            output.append(f'CODE:\n{row["code"]}')
    return '\n'.join(output) + '\n'


def run_baseline(input_path: Path = DATASET_PATH, output_root: Path = PROJECT_ROOT) -> dict:
    input_path, output_root = Path(input_path), Path(output_root)
    frame = load_dataset(input_path)
    parts = split_dataset(frame)
    train, validation, test = parts
    for name, part in zip(SPLIT_NAMES, parts):
        if set(part['label']) != {0, 1}:
            raise ValueError(f'{name} label must contain both classes for meaningful evaluation')
    print_summary(frame, parts)
    save_splits(parts, output_root / 'data/processed')
    comparison, best = [], None
    # Test data is never passed to the candidate fitting/selection loop.
    for ngram_range in NGRAM_RANGES:
        print(f'\nTraining character n-grams {ngram_range}...', flush=True)
        vectorizer, model = fit_candidate(train, ngram_range)
        metrics, _, _ = evaluate(vectorizer, model, validation)
        comparison.append({'ngram_range': list(ngram_range),
                           'precision': metrics['bug_precision'], 'recall': metrics['bug_recall'],
                           'f1': metrics['bug_f1']})
        print_metrics(f'VALIDATION RESULTS {ngram_range}', metrics)
        # Ties retain the first candidate in the documented comparison order.
        if best is None or metrics['bug_f1'] > best[2]['bug_f1']:
            best = (vectorizer, model, metrics, ngram_range)
    vectorizer, model, validation_metrics, selected = best
    print('\nngram_range | precision | recall | f1')
    for row in comparison:
        print(f'{tuple(row["ngram_range"])} | {row["precision"]:.4f} | '
              f'{row["recall"]:.4f} | {row["f1"]:.4f}')
    print(f'Selected ngram_range: {selected}')
    # Keep the selected train-only model, so no retraining or additional test selection occurs.
    test_metrics, predicted, probabilities = evaluate(vectorizer, model, test)
    print_metrics('FINAL TEST RESULTS', test_metrics)
    mutations = mutation_analysis(test, predicted)
    print('\nTest results by mutation type:')
    print(json.dumps(mutations, indent=2))
    examples = error_examples(test, predicted, probabilities)
    example_text = format_examples(examples)
    print(example_text)
    report = {
        'model': 'TF-IDF + Logistic Regression', 'random_state': 42,
        'ngram_range': list(selected),
        'accuracy': test_metrics['accuracy'], 'bug_precision': test_metrics['bug_precision'],
        'bug_recall': test_metrics['bug_recall'], 'bug_f1': test_metrics['bug_f1'],
        'dataset': dataset_statistics(frame),
        'input_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest(),
        'splits': {name: {'rows': len(part), 'unique_source_ids': int(part.source_id.nunique())}
                   for name, part in zip(SPLIT_NAMES, parts)},
        'source_id_overlap': check_source_overlap(*parts), 'comparison': comparison,
        'selection': 'Highest validation buggy F1; ties use first candidate; no refit',
        'vectorizer': {'analyzer': 'char', 'max_features': 30000, 'lowercase': False,
                       'sublinear_tf': True, 'fitted_features': len(vectorizer.vocabulary_)},
        'classifier': {'max_iter': 2000, 'class_weight': 'balanced', 'random_state': 42},
        'validation': validation_metrics, 'test': test_metrics,
        'mutation_analysis': mutations, 'error_examples': examples,
        'packages': {name: version(name) for name in ('pandas', 'scikit-learn', 'numpy', 'joblib')},
    }
    models, results = output_root / 'models', output_root / 'results'
    models.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, models / 'tfidf_vectorizer.joblib')
    joblib.dump(model, models / 'logistic_regression.joblib')
    (results / 'tfidf_logistic_regression_metrics.json').write_text(
        json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (results / 'tfidf_error_examples.txt').write_text(example_text, encoding='utf-8')
    predictions = test.copy()
    predictions['predicted_label'], predictions['bug_probability'] = predicted, probabilities
    predictions.to_csv(results / 'tfidf_test_predictions.csv', index=False)
    print(f'Saved models: {models}\nSaved results: {results}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DATASET_PATH)
    parser.add_argument('--output-root', type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    run_baseline(args.input, args.output_root)


if __name__ == '__main__':
    main()
