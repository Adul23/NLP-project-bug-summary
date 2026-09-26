"""End-to-end fixture tests; no dependency on the real held-out test data."""
import importlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest


def trainer():
    module = importlib.import_module('src.bug_detection.train_tfidf')
    assert hasattr(module, 'fit_candidate'), 'TF-IDF baseline is not implemented'
    return module


def data():
    return pd.DataFrame([
        {'sample_id': 2*i+y, 'source_id': f'{i:06d}',
         'code': f'def f{i}(x): return x {"+" if y == 0 else "-"} 1',
         'label': y, 'mutation_type': 'clean' if y == 0 else 'arithmetic_operator'}
        for i in range(40) for y in (0, 1)
    ])


def test_train_only_vocabulary_predictions_and_probabilities():
    vectorizer, model = trainer().fit_candidate(data(), (3, 5))
    held_out = ['UNSEEN_SENTINEL_zzz']
    before = dict(vectorizer.vocabulary_)
    transformed = vectorizer.transform(held_out)
    assert vectorizer.vocabulary_ == before
    assert 'zzz' not in before
    assert transformed.shape[0] == 1
    assert model.predict(transformed).shape == (1,)
    probability = model.predict_proba(transformed)
    assert np.all((probability >= 0) & (probability <= 1))
    assert vectorizer.analyzer == 'char'
    assert not vectorizer.lowercase


def test_complete_run_saves_metrics_models_splits_and_inference(tmp_path, monkeypatch):
    module = trainer()
    path = tmp_path / 'data/processed/bug_detection_dataset.csv'
    path.parent.mkdir(parents=True)
    data().to_csv(path, index=False)
    report = module.run_baseline(path, tmp_path)
    saved = json.loads((tmp_path / 'results/tfidf_logistic_regression_metrics.json').read_text())
    assert report == saved
    assert len(report['comparison']) == 3
    assert report['bug_f1'] == report['test']['bug_f1']
    assert report['validation']['bug_f1'] == max(r['f1'] for r in report['comparison'])
    assert np.asarray(report['test']['confusion_matrix']).sum() == report['splits']['test']['rows']
    assert report['source_id_overlap'] == {'train/validation': 0, 'train/test': 0, 'validation/test': 0}
    assert report['mutation_analysis']['boolean_operator']['recall'] is None
    for name in ('train', 'validation', 'test'):
        assert (path.parent / f'{name}.csv').exists()
    predictor = importlib.import_module('src.bug_detection.predict_tfidf')
    monkeypatch.setattr(predictor, 'MODEL_DIR', tmp_path / 'models')
    result = predictor.predict_bug('def f(x): return x - 1')
    assert set(result) == {'label', 'bug_probability'}
    assert result['label'] in {'clean', 'buggy'}
    assert 0 <= result['bug_probability'] <= 1
    vectorizer = joblib.load(tmp_path / 'models/tfidf_vectorizer.joblib')
    model = joblib.load(tmp_path / 'models/logistic_regression.joblib')
    expected = model.predict_proba(vectorizer.transform(['def f(x): return x - 1']))[0, 1]
    assert result['bug_probability'] == pytest.approx(expected)
    with pytest.raises(ValueError, match='code'):
        predictor.predict_bug(' ')

def test_selection_uses_validation_and_test_is_evaluated_once(tmp_path, monkeypatch):
    """Catch held-out fitting, test-based selection, or repeated test evaluation."""
    module = trainer()
    input_frame = data()
    path = tmp_path / 'input.csv'
    input_frame.to_csv(path, index=False)
    expected_train, expected_val, expected_test = module.split_dataset(input_frame)
    train_ids = set(expected_train.source_id)
    val_ids = set(expected_val.source_id)
    test_ids = set(expected_test.source_id)
    real_fit, real_evaluate = module.fit_candidate, module.evaluate
    fitted, calls = [], []
    # Controlled validation scores isolate selection from accidental fixture ties.
    validation_scores = iter([0.2, 0.8, 0.4])

    def observed_fit(frame, ngram_range):
        assert set(frame.source_id) == train_ids
        pair = real_fit(frame, ngram_range)
        fitted.append(pair)
        return pair

    def observed_evaluate(vectorizer, model, frame):
        ids = set(frame.source_id)
        metrics, predicted, probabilities = real_evaluate(vectorizer, model, frame)
        if ids == val_ids:
            assert 'test' not in calls
            calls.append('validation')
            metrics['bug_f1'] = next(validation_scores)
        else:
            assert ids == test_ids
            assert calls == ['validation', 'validation', 'validation']
            assert vectorizer is fitted[1][0]
            assert model is fitted[1][1]
            calls.append('test')
        return metrics, predicted, probabilities

    monkeypatch.setattr(module, 'fit_candidate', observed_fit)
    monkeypatch.setattr(module, 'evaluate', observed_evaluate)
    report = module.run_baseline(path, tmp_path)
    assert len(fitted) == 3
    assert calls == ['validation', 'validation', 'validation', 'test']
    assert report['ngram_range'] == [3, 5]
