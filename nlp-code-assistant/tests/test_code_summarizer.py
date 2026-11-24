"""Fine-tuned CodeT5 summary feature: health, /summarize, /analyze summarize flag, CLI.

CodeSummarizer is replaced by fakes; no real weights are loaded here.
"""

import json
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from src.api import app as app_module
from src.api.app import create_app
from src.bug_detection.experiment_data import create_splits
from src.bug_detection.train_comparison_tfidf import train as train_tfidf
from src.inference import cli
from src.inference.models import CodeSummarizer

CODE = "def add(a, b):\n    return a + b"


class FakeSummarizer:
    instances = 0

    def __init__(self, folder):
        FakeSummarizer.instances += 1
        self.folder = folder

    def summarize(self, code):
        return {
            "summary": f"Summarize {len(code)} characters.",
            "summary_source": "codet5_finetuned",
            "model": "codet5_summarizer",
        }


class BrokenSummarizer(FakeSummarizer):
    def summarize(self, code):
        raise RuntimeError("generation failed")


class UnloadableSummarizer:
    def __init__(self, folder):
        raise OSError("corrupt weights")


def fake_best(root):
    best = root / "codet5_summarizer" / "best"
    best.mkdir(parents=True)
    (best / "config.json").write_text("{}")
    (best / "model.safetensors").write_bytes(b"not real weights")
    return best


@pytest.fixture(scope="module")
def tfidf_root(tmp_path_factory):
    """A models/ folder with a real (tiny) TF-IDF model, so /analyze can predict."""
    base = tmp_path_factory.mktemp("summary_api")
    rows = []
    for i in range(60):
        for label in [0, 1] if i < 40 else [0]:
            rows.append(
                {
                    "sample_id": len(rows) + 1,
                    "source_id": f"{i:06d}",
                    "code": f"def f{i}(x): return x {'+' if label == 0 else '-'} 1",
                    "label": label,
                    "mutation_type": "clean" if label == 0 else "arithmetic_operator",
                }
            )
    pd.DataFrame(rows).to_csv(base / "dataset.csv", index=False)
    create_splits(base / "dataset.csv", base / "splits")
    train_tfidf(base / "splits", base / "tfidf_model")
    return base / "tfidf_model"


@pytest.fixture
def models(tmp_path, tfidf_root):
    root = tmp_path / "models"
    root.mkdir()
    (root / "tfidf_v2").symlink_to(tfidf_root, target_is_directory=True)
    return root


def client_for(models, tmp_path):
    return TestClient(create_app(models, tmp_path / "feedback"))


def test_health_reports_summarizer_states(models, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "CodeSummarizer", FakeSummarizer)
    with client_for(models, tmp_path) as client:
        status = client.get("/health").json()["models"]
        assert status["code_summarizer"]["status"] == "unavailable"
        # The LoRA explainer is a separate model and stays unavailable.
        assert status["summarizer"]["status"] == "unavailable"

    best = fake_best(models)
    with client_for(models, tmp_path) as client:
        status = client.get("/health").json()["models"]
        assert status["code_summarizer"]["status"] == "artifacts_present_not_loaded"
        assert status["summarizer"]["status"] == "unavailable"
        assert client.post("/summarize", json={"code": CODE}).status_code == 200
        assert client.get("/health").json()["models"]["code_summarizer"]["status"] == "loaded"

    (best / "model.safetensors").unlink()  # config alone is not enough
    with client_for(models, tmp_path) as client:
        assert client.get("/health").json()["models"]["code_summarizer"]["status"] == "unavailable"


def test_summarize_endpoint(models, tmp_path, monkeypatch):
    fake_best(models)
    monkeypatch.setattr(app_module, "CodeSummarizer", FakeSummarizer)
    FakeSummarizer.instances = 0
    with client_for(models, tmp_path) as client:
        result = client.post("/summarize", json={"code": CODE}).json()
        assert result == {
            "summary": f"Summarize {len(CODE)} characters.",
            "summary_source": "codet5_finetuned",
            "model": "codet5_summarizer",
        }
        client.post("/summarize", json={"code": CODE})
        assert FakeSummarizer.instances == 1  # loaded once, then cached
        for payload in [
            {"code": ""},
            {"code": "   "},
            {"code": 1},
            {"code": "x" * 20001},
            {"code": "x", "model": "tfidf"},  # no model field on /summarize
        ]:
            assert client.post("/summarize", json=payload).status_code == 422


def test_summarize_503_when_missing_or_broken(models, tmp_path, monkeypatch):
    # Real CodeSummarizer, no best/ folder: fails before loading anything.
    with client_for(models, tmp_path) as client:
        response = client.post("/summarize", json={"code": CODE})
        assert response.status_code == 503
        assert "code_summarizer unavailable" in response.json()["detail"]
        assert client.get("/health").json()["models"]["code_summarizer"]["status"] == "load_failed"

    fake_best(models)
    monkeypatch.setattr(app_module, "CodeSummarizer", UnloadableSummarizer)
    with client_for(models, tmp_path) as client:
        assert client.post("/summarize", json={"code": CODE}).status_code == 503

    monkeypatch.setattr(app_module, "CodeSummarizer", BrokenSummarizer)
    with client_for(models, tmp_path) as client:
        assert client.post("/summarize", json={"code": CODE}).status_code == 503


def test_code_summarizer_checks_files_before_importing_transformers(tmp_path):
    with pytest.raises(FileNotFoundError, match="config.json"):
        CodeSummarizer(tmp_path / "codet5_summarizer")


@pytest.mark.parametrize(
    "summarizer, expected_summary, expected_source",
    [
        (FakeSummarizer, f"Summarize {len(CODE)} characters.", "codet5_finetuned"),
        (BrokenSummarizer, None, "unavailable"),
        (UnloadableSummarizer, None, "unavailable"),
    ],
)
def test_analyze_with_summary(models, tmp_path, monkeypatch, summarizer, expected_summary, expected_source):
    fake_best(models)
    monkeypatch.setattr(app_module, "CodeSummarizer", summarizer)
    with client_for(models, tmp_path) as client:
        response = client.post("/analyze", json={"code": CODE, "summarize": True})
        assert response.status_code == 200  # a failing summarizer never fails the analysis
        result = response.json()
        assert result["label"] in ("BUGGY", "CLEAN") and result["model"] == "tfidf"
        assert 0 <= result["bug_probability"] <= 1
        assert result["summary"] == expected_summary
        assert result["summary_source"] == expected_source
        # The explanation path is untouched by the summary.
        assert result["explanation_source"] == "unavailable"


def test_analyze_summary_missing_model_and_default_off(models, tmp_path, monkeypatch):
    with client_for(models, tmp_path) as client:
        # No best/ folder at all, real class: still a successful analysis.
        result = client.post("/analyze", json={"code": CODE, "summarize": True}).json()
        assert result["summary"] is None and result["summary_source"] == "unavailable"
        # Off by default: no summary keys and the summarizer is not touched.
        plain = client.post("/analyze", json={"code": CODE}).json()
        assert "summary" not in plain and "summary_source" not in plain
        assert client.post("/analyze", json={"code": CODE, "summarize": "yes"}).status_code == 422
        # The LoRA explainer path still reports itself unavailable.
        assert client.post("/analyze", json={"code": CODE, "explanation": "lora"}).status_code == 503


def test_cli_code_summarizer(tmp_path, monkeypatch, capsys):
    seen = {}

    class CliFake(FakeSummarizer):
        def __init__(self, folder):
            seen["folder"] = folder

    monkeypatch.setattr(cli, "CodeSummarizer", CliFake)
    source = tmp_path / "add.py"
    source.write_text(CODE)
    monkeypatch.setattr(sys, "argv", ["cli", "--model", "code_summarizer", "--file", str(source)])
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result["summary_source"] == "codet5_finetuned" and result["model"] == "codet5_summarizer"
    assert seen["folder"].name == "codet5_summarizer"  # default models/codet5_summarizer

    monkeypatch.setattr(
        sys, "argv", ["cli", "--model", "code_summarizer", "--file", str(source), "--model-dir", str(tmp_path / "x")]
    )
    cli.main()
    assert seen["folder"] == tmp_path / "x"

    monkeypatch.setattr(cli, "CodeSummarizer", UnloadableSummarizer)
    monkeypatch.setattr(sys, "argv", ["cli", "--model", "code_summarizer", "--file", str(source)])
    with pytest.raises(SystemExit) as stop:
        cli.main()
    assert stop.value.code == 1
    assert "Model unavailable" in capsys.readouterr().err
