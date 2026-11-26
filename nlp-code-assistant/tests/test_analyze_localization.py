"""/analyze without original code: operator-flip localization as an unverified explanation.

TfidfPredictor is replaced by a fake whose bug probability depends on the code text.
"""

import warnings

import pytest
from fastapi.testclient import TestClient
from src.api import app as app_module
from src.api.app import create_app
from src.bug_detection.localize import repair_candidates

BUGGY = "def f(a, b):\n    if a < b:\n        return a - b\n    return 0\n"
FIXED = BUGGY.replace("a < b", "a > b")


class FakePredictor:
    base = 0.9
    # substring of a candidate -> its bug probability; anything else scores `base`
    scores = {"a > b": 0.1, "a <= b": 0.5, "a + b": 0.7}

    def __init__(self, folder):
        pass

    def predict_batch(self, codes):
        out = []
        for code in codes:
            p = next((v for k, v in self.scores.items() if k in code), self.base)
            out.append(
                {
                    "label": "BUGGY" if p >= 0.5 else "CLEAN",
                    "bug_probability": p,
                    "confidence": p if p >= 0.5 else 1 - p,
                    "model": "tfidf",
                }
            )
        return out

    def predict(self, code):
        return self.predict_batch([code])[0]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "TfidfPredictor", FakePredictor)
    with TestClient(create_app(tmp_path / "models", tmp_path / "feedback")) as c:
        yield c


def analyze(client, **body):
    response = client.post("/analyze", json={"model": "tfidf", **body})
    assert response.status_code == 200, response.text
    return response.json()


def test_buggy_code_gets_operator_flip_explanation(client):
    result = analyze(client, code=BUGGY)
    assert result["explanation_source"] == "operator_flip"
    assert result["verified"] is False
    assert result["suspected_line"] == 2
    assert result["explanation"] == "Line 2: `<` may be `>`"
    assert result["bug_type"] == "comparison_operator"
    assert result["suggested_fix"] == FIXED
    candidates = result["repair_candidates"]
    assert [(c["line"], c["old"], c["new"]) for c in candidates] == [
        (2, "<", ">"),
        (2, "<", "<="),
        (3, "-", "+"),
    ]
    assert candidates[0]["drop"] == pytest.approx(0.8)
    assert candidates[0]["mutation_type"] == "comparison_operator"


def test_original_code_keeps_verified_explanation(client):
    result = analyze(client, code=BUGGY, original_code=FIXED)
    assert result["explanation_source"] == "verified_code_pair"
    assert result["verified"] is True
    assert "suspected_line" not in result
    assert "repair_candidates" not in result


def test_low_probability_gets_no_explanation(client, monkeypatch):
    monkeypatch.setattr(FakePredictor, "base", 0.3)
    result = analyze(client, code=BUGGY)
    assert result["explanation_source"] == "unavailable"
    assert result["explanation"] is None
    assert "suspected_line" not in result


def test_no_candidate_lowers_probability(client, monkeypatch):
    monkeypatch.setattr(FakePredictor, "scores", {})
    result = analyze(client, code=BUGGY)
    assert result["explanation_source"] == "unavailable"
    assert result["suggested_fix"] is None


def test_invalid_syntax_is_not_an_error(client):
    result = analyze(client, code="def f(a, b):\n    return a <\n")
    assert result["explanation_source"] == "unavailable"
    assert result["label"] == "BUGGY"


def test_invalid_escape_does_not_warn():
    code = 'def f(s):\n    return s == "\\d" and s < "x"\n'
    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        assert repair_candidates(code)  # would raise if ast.parse warned
