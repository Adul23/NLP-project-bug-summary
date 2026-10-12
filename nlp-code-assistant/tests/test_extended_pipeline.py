import json
import subprocess
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from src.api.app import create_app
from src.bug_detection.experiment_data import create_splits, load_splits
from src.bug_detection.train_comparison_tfidf import train
from src.summarization.prepare import prepare, verified_change
from src.summarization.train_lora import load_supervision


@pytest.fixture
def shared(tmp_path):
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
    dataset = tmp_path / "dataset.csv"
    pd.DataFrame(rows).to_csv(dataset, index=False)
    splits = tmp_path / "splits"
    create_splits(dataset, splits)
    return splits


def test_stratification_and_manifest(shared):
    parts, manifest = load_splits(shared)
    assert [len(p) for p in parts] == [70, 15, 15]
    assert [p.label.sum() for p in parts] == [28, 6, 6]
    assert not any(manifest["source_id_overlap"].values())
    with (shared / "test.csv").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="manifest"):
        load_splits(shared)


@pytest.mark.parametrize(
    "old,new,kind",
    [
        ("==", "!=", "comparison_operator"),
        ("+", "-", "arithmetic_operator"),
        ("and", "or", "boolean_operator"),
    ],
)
def test_verified_change(old, new, kind):
    result = verified_change(
        f"def f(a,b): return a {old} b", f"def f(a,b): return a {new} b"
    )
    assert result["mutation_type"] == kind
    assert result["old_operator"] == old and result["new_operator"] == new
    assert result["suggested_fix"] is None
    assert result["explanation"].startswith("Synthetic mutation:")


@pytest.mark.parametrize(
    "code",
    [
        "def f(a,b): return a + b",
        "def f(a,b): return a - 2",
        "bad syntax",
        'def f(a,b): return "-"',
    ],
)
def test_unverified_changes_rejected(code):
    with pytest.raises(ValueError):
        verified_change("def f(a,b): return a + b", code)


def test_pair_supervision(shared, tmp_path):
    output = tmp_path / "summaries"
    assert prepare(shared, output) == {"train": 28, "validation": 6, "test": 6}
    parts = load_supervision(output)
    sources, _ = load_splits(shared)
    for rows, source in zip(parts, sources):
        assert {r["source_id"] for r in rows} <= set(source.source_id)
    parts[0][0]["explanation"] = "This definitely crashes."
    (output / "train.jsonl").write_text("\n".join(json.dumps(r) for r in parts[0]))
    with pytest.raises(ValueError, match="explanation"):
        load_supervision(output)


@pytest.fixture
def api(shared, tmp_path):
    root = tmp_path / "models"
    assert train(shared, root / "tfidf_v2")["reload_predictions_identical"]
    with TestClient(create_app(root)) as client:
        yield client


def test_api_prediction_and_evidence(api):
    assert (
        api.get("/health").json()["models"]["tfidf"]["status"]
        == "artifacts_present_not_loaded"
    )
    code = "def f(x): return x - 1"
    result = api.post("/predict", json={"code": code}).json()
    assert result["label"] in ["CLEAN", "BUGGY"] and result["model"] == "tfidf"
    assert 0 <= result["bug_probability"] <= 1
    assert result["confidence"] == (
        result["bug_probability"]
        if result["label"] == "BUGGY"
        else 1 - result["bug_probability"]
    )
    assert api.get("/health").json()["models"]["tfidf"]["status"] == "loaded"
    plain = api.post("/analyze", json={"code": code}).json()
    assert plain["bug_type"] is plain["explanation"] is plain["suggested_fix"] is None
    paired = api.post(
        "/analyze", json={"code": code, "original_code": "def f(x): return x + 1"}
    ).json()
    assert paired["verified"] and paired["bug_type"] == "arithmetic_operator_change"
    assert paired["suggested_fix"] is None


def test_api_limits_errors(api):
    for payload in [
        {"code": ""},
        {"code": " "},
        {"code": 2},
        {"code": "x", "model": "unknown"},
        {"code": "x" * 20001},
        {"code": "x", "metadata": "invented"},
    ]:
        assert api.post("/predict", json=payload).status_code == 422
    assert api.post("/predict", content=b"x" * 128001).status_code == 413
    assert (
        api.post("/predict", json={"code": "x", "model": "codebert"}).status_code == 503
    )
    assert (
        api.post("/analyze", json={"code": "x", "explanation": "lora"}).status_code
        == 503
    )
    assert (
        api.post(
            "/analyze",
            json={"code": "def f(): return 1", "original_code": "def f(): return 2"},
        ).status_code
        == 422
    )


def test_imports_do_not_train_or_import_transformers():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import src.api.app; import src.bug_detection.train_lora; import src.summarization.train_lora; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules",
        ],
        check=True,
    )


def test_prepare_refuses_overwriting_data(shared, tmp_path):
    destination = tmp_path / "supervision"
    prepare(shared, destination)
    original = (destination / "train.jsonl").read_bytes()
    with pytest.raises(ValueError, match="empty"):
        prepare(shared, destination)
    assert (destination / "train.jsonl").read_bytes() == original


def test_api_never_executes_submitted_code(api, tmp_path):
    marker = tmp_path / "should_not_exist"
    code = f"def f(): return __import__('pathlib').Path({str(marker)!r}).touch()"
    assert api.post("/predict", json={"code": code}).status_code == 200
    assert not marker.exists()


def test_streamed_body_limit():
    import asyncio

    from src.api.app import BodyLimit

    messages = iter(
        [
            {"type": "http.request", "body": b"x" * 70000, "more_body": True},
            {"type": "http.request", "body": b"x" * 70000, "more_body": False},
        ]
    )
    responses = []

    async def receive():
        return next(messages)

    async def send(message):
        responses.append(message)

    async def downstream(*args):
        raise AssertionError("Oversized request reached application")

    asyncio.run(BodyLimit(downstream)({"type": "http"}, receive, send))
    assert responses[0]["status"] == 413
