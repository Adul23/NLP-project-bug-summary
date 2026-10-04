"""Predict CLEAN/BUGGY from source text using the saved baseline artifacts."""
from pathlib import Path

import joblib

MODEL_DIR = Path(__file__).resolve().parents[2] / 'models'


def predict_bug(code: str) -> dict:
    if not isinstance(code, str) or not code.strip():
        raise ValueError('code must be a nonempty string')
    vectorizer_path = MODEL_DIR / 'tfidf_vectorizer.joblib'
    model_path = MODEL_DIR / 'logistic_regression.joblib'
    if not vectorizer_path.is_file() or not model_path.is_file():
        raise FileNotFoundError('Saved baseline models are missing; run src/bug_detection/train_tfidf.py first')
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    matrix = vectorizer.transform([code])
    label = int(model.predict(matrix)[0])
    probability = float(model.predict_proba(matrix)[0, list(model.classes_).index(1)])
    return {'label': 'buggy' if label == 1 else 'clean', 'bug_probability': probability}
