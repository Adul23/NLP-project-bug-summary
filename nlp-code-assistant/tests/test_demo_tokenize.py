"""TF-IDF part of scripts/demo_tokenize.py on a tiny model fitted here (no real artifacts)."""

import importlib.util
import io
from pathlib import Path

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.inference.models import TfidfPredictor

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/demo_tokenize.py"
spec = importlib.util.spec_from_file_location("demo_tokenize", SCRIPT)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)

CODE = "def f(a, b):\n    return a < b\n"


@pytest.fixture
def model_dir(tmp_path):
    codes = [
        "def f(a, b):\n    return a > b\n",
        "def g(x):\n    return x + 1\n",
        "def f(a, b):\n    return a < b\n",
        "def g(x):\n    return x - 1\n",
    ]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), lowercase=False)
    model = LogisticRegression().fit(vectorizer.fit_transform(codes), [0, 0, 1, 1])
    joblib.dump(vectorizer, tmp_path / "tfidf_vectorizer.joblib")
    joblib.dump(model, tmp_path / "logistic_regression.joblib")
    return tmp_path


def test_tfidf_report_explains_the_probability(model_dir):
    predictor = TfidfPredictor(model_dir)
    report = demo.tfidf_report(CODE, predictor, top=5)
    assert report["nonzero_features"] == predictor.vectorizer.transform([CODE]).nnz
    assert len(report["top_ngrams"]) == 5
    sizes = [abs(row["contribution"]) for row in report["top_ngrams"]]
    assert sizes == sorted(sizes, reverse=True)
    # contributions + intercept reproduce the classifier's own probability
    assert report["sigmoid_of_logit"] == pytest.approx(report["bug_probability"])
    loc = report["localize"]
    assert loc["n_candidates"] >= 1
    assert {c["old"] for c in loc["candidates"]} == {"<"}


def test_cli_reads_stdin(model_dir, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(CODE))
    report = demo.main(["--model", "tfidf", "--model-dir", str(model_dir)])
    out = capsys.readouterr().out
    assert "non-zero of" in out and "Prediction:" in out and "line 2: `<`" in out
    assert report["label"] in ("BUGGY", "CLEAN")


def test_unparsable_code_is_reported_not_raised(model_dir):
    report = demo.tfidf_report("def f(:\n", TfidfPredictor(model_dir))
    assert "cannot parse" in report["localize"]["error"]


def test_empty_input_is_rejected(tmp_path):
    empty = tmp_path / "empty.py"
    empty.write_text("  \n")
    with pytest.raises(ValueError):
        demo.read_code(empty)
