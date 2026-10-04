"""Predict with the persisted character TF-IDF baseline."""
from pathlib import Path

import joblib
import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


def predict_bug(code: str) -> dict:
    """Return a clean/buggy label and the classifier's probability for class 1."""
    if not isinstance(code, str):
        raise TypeError("code must be a string")
    if not code.strip():
        raise ValueError("code must not be empty")
    vectorizer_path = MODEL_DIR / "tfidf_vectorizer.joblib"
    model_path = MODEL_DIR / "logistic_regression.joblib"
    if not vectorizer_path.exists() or not model_path.exists():
        raise FileNotFoundError("Train the baseline first: python src/bug_detection/train_tfidf.py")
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    features = vectorizer.transform([code])
    prediction = int(model.predict(features)[0])
    bug_column = int(np.flatnonzero(model.classes_ == 1)[0])
    probability = float(model.predict_proba(features)[0, bug_column])
    return {"label": "buggy" if prediction else "clean", "bug_probability": probability}
