"""Training-pipeline optimizations: precision, tokenization, replay, swap gate, retrain jobs."""

import json
import threading
import time
from argparse import Namespace
from datetime import UTC, datetime

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from src.api.app import create_app
from src.bug_detection import retrain_from_feedback as rf
from src.bug_detection.lora_common import class_weights, select_precision

# --- precision selection (pure logic, no torch needed) --------------------------------


@pytest.mark.parametrize(
    "requested, cuda, bf16, expected",
    [
        ("auto", False, False, "fp32"),  # CPU never uses mixed precision
        ("auto", True, True, "bf16"),  # Ampere and newer
        ("auto", True, False, "fp16"),  # T4: no native bf16 -> fp16, not fp32
        ("fp32", True, True, "fp32"),
        ("fp16", True, True, "fp16"),
        ("bf16", True, True, "bf16"),
        ("fp32", False, False, "fp32"),
    ],
)
def test_precision_selection(requested, cuda, bf16, expected):
    assert select_precision(requested, cuda, bf16) == expected


def test_precision_selection_rejects_unsupported():
    with pytest.raises(ValueError, match="requires CUDA"):
        select_precision("fp16", False, False)
    with pytest.raises(ValueError, match="requires CUDA"):
        select_precision("bf16", False, False)
    with pytest.raises(ValueError, match="no native bf16"):
        select_precision("bf16", True, False)
    with pytest.raises(ValueError):
        select_precision("fp8", True, True)
    # The summarizer keeps fp32 under auto when bf16 is missing (T5 can overflow in fp16).
    assert select_precision("auto", True, False, allow_fp16_auto=False) == "fp32"
    assert select_precision("fp16", True, False, allow_fp16_auto=False) == "fp16"


def test_class_weights_balance_the_imbalance():
    clean, buggy = class_weights([0] * 7000 + [1] * 3320)
    assert clean == pytest.approx(10320 / 14000) and buggy == pytest.approx(10320 / 6640)
    assert clean * 7000 == pytest.approx(buggy * 3320)  # equal total weight per class
    with pytest.raises(ValueError):
        class_weights([0, 0])


# --- replay mixing, gate and versioned folders (no torch needed) ----------------------


def train_frame(clean=70, buggy=30):
    rows = [(f"def c{i}(x): return x + {i}", 0) for i in range(clean)]
    rows += [(f"def b{i}(x): return x - {i}", 1) for i in range(buggy)]
    return pd.DataFrame(rows, columns=["code", "label"])


def test_replay_mixing_ratio_and_stratification():
    feedback = [{"code": f"def fb{i}(x): return x", "label": i % 2} for i in range(5)]
    feedback.append({"code": "def c0(x): return x + 0", "label": 1})  # also in train
    records, mix = rf.build_training_mix(
        feedback, train_frame(), replay_ratio=4, feedback_oversample=2
    )
    replay = [r for r in records if r["source"] == "replay"]
    from_feedback = [r for r in records if r["source"] == "feedback"]
    assert len(replay) == 4 * len(feedback) == mix["replay_rows"] == 24
    assert len(from_feedback) == 2 * len(feedback)  # every feedback row oversampled x2
    assert mix["total_rows"] == len(records) == 36
    # Label-stratified like the train split (30% buggy), and never the feedback's own code.
    assert sum(r["label"] for r in replay) == round(24 * 30 / 99)
    assert "def c0(x): return x + 0" not in {r["code"] for r in replay}
    assert len({r["code"] for r in replay}) == len(replay)
    # Deterministic, and a ratio of 0 disables replay.
    assert rf.build_training_mix(feedback, train_frame())[0] == records
    assert rf.build_training_mix(feedback, train_frame(), replay_ratio=0)[1]["replay_rows"] == 0
    # Never more replay rows than the split can provide (c0 is excluded: it is in the feedback).
    assert rf.build_training_mix(feedback, train_frame(4, 2), replay_ratio=50)[1]["replay_rows"] == 5
    with pytest.raises(ValueError):
        rf.build_training_mix([], train_frame())


def test_gate_decision_threshold():
    assert rf.gate_decision(0.75, 0.80)
    assert rf.gate_decision(0.75, 0.75)
    assert rf.gate_decision(0.75, 0.74)  # a drop of exactly 0.01 is tolerated
    assert not rf.gate_decision(0.75, 0.7399)
    assert not rf.gate_decision(0.75, 0.50)
    assert rf.gate_decision(0.75, 0.70, max_drop=0.05)


def test_versioned_dirs_and_active_pointer(tmp_path):
    stamp = datetime(2026, 11, 14, 9, 30, 5, tzinfo=UTC)
    assert rf.read_active_dir(tmp_path) == tmp_path / "codebert_lora"  # no pointer yet
    first = rf.next_version_dir(tmp_path, stamp)
    assert first.name == "codebert_lora_v2_20261114T093005Z"  # the original is v1
    (tmp_path / "codebert_lora_v2").mkdir()  # legacy un-stamped folder from the old endpoint
    (tmp_path / "codebert_lora_v7_20260101T000000Z").mkdir()
    (tmp_path / "codebert_lora_v9_notes.txt").write_text("not a folder")
    assert rf.next_version_dir(tmp_path, stamp).name == "codebert_lora_v8_20261114T093005Z"

    target = tmp_path / "codebert_lora_v7_20260101T000000Z"
    rf.write_active_pointer(tmp_path, target, previous=tmp_path / "codebert_lora")
    # A pointer to a folder without artifacts is ignored, so a broken swap cannot brick the API.
    assert rf.read_active_dir(tmp_path) == tmp_path / "codebert_lora"
    (target / "training_config.json").write_text("{}")
    assert rf.read_active_dir(tmp_path) == target
    pointer = json.loads((tmp_path / rf.ACTIVE_POINTER).read_text())
    assert pointer["active_dir"] == target.name  # relative, so models/ can be moved
    (tmp_path / rf.ACTIVE_POINTER).write_text("{broken")
    assert rf.read_active_dir(tmp_path) == tmp_path / "codebert_lora"


# --- retrain job endpoints (training function mocked) ---------------------------------

METRICS = {"accuracy": 0.8, "bug_precision": 0.7, "bug_recall": 0.7, "bug_f1": 0.7}


@pytest.fixture
def job_api(tmp_path, monkeypatch):
    models, feedback = tmp_path / "models", tmp_path / "feedback"
    for name in ("codebert_lora", "codebert_lora_v2_20260101T000000Z"):
        (models / name / "adapter").mkdir(parents=True)
        (models / name / "training_config.json").write_text('{"status": "trained"}')
    feedback.mkdir()
    (feedback / "user_feedback.jsonl").write_text(
        json.dumps({"code": "def f(a, b):\n    return a - b", "actual_label": "BUGGY", "is_mistake": True})
        + "\n"
    )
    state = {"release": threading.Event(), "calls": [], "outcome": "accepted"}

    def fake_retrain(**kwargs):
        state["calls"].append(kwargs)
        kwargs["progress"](phase="training", epoch=1, epochs=kwargs["epochs"], loss=0.5)
        assert state["release"].wait(10)
        if state["outcome"] == "error":
            raise ValueError("Split dir is missing")
        new = dict(METRICS, bug_f1=0.72 if state["outcome"] == "accepted" else 0.4)
        out = models / "codebert_lora_v3_20261114T000000Z"
        (out / "adapter").mkdir(parents=True, exist_ok=True)
        (out / "training_config.json").write_text('{"status": "trained"}')
        return {
            "status": state["outcome"],
            "output_dir": str(out),
            "active_model_dir": str(out),
            "old_metrics": METRICS,
            "new_metrics": new,
        }

    monkeypatch.setattr(rf, "retrain_lora", fake_retrain)
    with TestClient(create_app(models, feedback)) as client:
        yield client, state, models


def wait_for(client, job_id, statuses=("done", "failed", "rejected")):
    for _ in range(200):
        job = client.get(f"/feedback/retrain/{job_id}").json()
        if job["status"] in statuses:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job stuck in {job['status']}")


def test_retrain_job_lifecycle_and_conflict(job_api):
    client, state, models = job_api
    assert client.get("/feedback/retrain").json() == {"job": None}
    started = client.post("/feedback/retrain", json={"epochs": 2, "mistakes_only": True})
    assert started.status_code == 202  # returns immediately, training runs in the background
    job_id = started.json()["job_id"]
    assert started.json()["status"] == "queued"

    running = wait_for(client, job_id, ("running",))
    for _ in range(100):  # progress arrives from the worker thread
        running = client.get(f"/feedback/retrain/{job_id}").json()
        if running["progress"].get("epoch"):
            break
        time.sleep(0.02)
    assert running["progress"] == {"phase": "training", "epoch": 1, "epochs": 2, "loss": 0.5}
    assert running["metrics"] is None

    conflict = client.post("/feedback/retrain", json={"epochs": 1, "mistakes_only": False})
    assert conflict.status_code == 409  # single-job lock
    assert len(state["calls"]) == 1

    state["release"].set()
    done = wait_for(client, job_id)
    assert done["status"] == "done" and done["result"] == "accepted"
    assert done["metrics"] == {"old": METRICS, "new": dict(METRICS, bug_f1=0.72)}
    assert done["active_model_dir"].endswith("codebert_lora_v3_20261114T000000Z")
    assert done["finished_at"] and done["error"] is None
    assert client.get("/feedback/retrain").json()["job"]["job_id"] == job_id
    call = state["calls"][0]
    assert call["epochs"] == 2 and call["mistakes_only"] is True
    assert call["base_lora_dir"] == models / "codebert_lora" and call["model_root"] == models

    # The hot swap happened: the next job continues from the accepted adapter.
    again = client.post("/feedback/retrain", json={"epochs": 1, "mistakes_only": False})
    assert again.status_code == 202
    wait_for(client, again.json()["job_id"])
    assert state["calls"][1]["base_lora_dir"].name == "codebert_lora_v3_20261114T000000Z"


def test_retrain_job_rejected_failed_and_unknown(job_api):
    client, state, models = job_api
    state["release"].set()
    state["outcome"] = "rejected_worse"
    job_id = client.post("/feedback/retrain", json={"epochs": 1, "mistakes_only": False}).json()["job_id"]
    rejected = wait_for(client, job_id)
    assert rejected["status"] == "rejected" and rejected["result"] == "rejected_worse"
    assert rejected["metrics"]["new"]["bug_f1"] == 0.4
    assert rejected["active_model_dir"] == str(models / "codebert_lora")  # not swapped

    state["outcome"] = "error"
    job_id = client.post("/feedback/retrain", json={"epochs": 1, "mistakes_only": False}).json()["job_id"]
    failed = wait_for(client, job_id)
    assert failed["status"] == "failed" and failed["error"] == "Split dir is missing"

    assert client.get("/feedback/retrain/nope").status_code == 404
    assert client.post("/feedback/retrain", json={"epochs": 11}).status_code == 422


def test_retrain_validates_before_starting_a_job(job_api, tmp_path):
    client, state, models = job_api
    (tmp_path / "feedback" / "user_feedback.jsonl").write_text("")
    response = client.post("/feedback/retrain", json={"epochs": 1, "mistakes_only": False})
    assert response.status_code == 422 and not state["calls"]
    assert client.get("/feedback/retrain").json() == {"job": None}


def test_api_reads_active_pointer_at_startup(job_api, tmp_path):
    _, state, models = job_api
    rf.write_active_pointer(models, models / "codebert_lora_v2_20260101T000000Z")
    state["release"].set()
    with TestClient(create_app(models, tmp_path / "feedback")) as client:
        job_id = client.post("/feedback/retrain", json={"epochs": 1, "mistakes_only": False}).json()["job_id"]
        wait_for(client, job_id)
    assert state["calls"][-1]["base_lora_dir"] == models / "codebert_lora_v2_20260101T000000Z"


# --- tests below need the optional transformer stack -----------------------------------

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
from src.bug_detection.lora_common import (  # noqa: E402
    classifier_target_modules,
    tokenize_once,
    tune_threshold,
)
from test_lora_local import tiny_base  # noqa: E402
from tokenizers.processors import TemplateProcessing  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

TEXTS = ["def x", "def x return x + 1 - x", "return " * 30, "x + 1 " * 4]


@pytest.mark.parametrize("specials", [False, True])
@pytest.mark.parametrize("side", ["right", "left"])
def test_single_pass_tokenization_matches_tokenizer_truncation(tmp_path, specials, side):
    tiny_base(tmp_path / "base")
    tokenizer = AutoTokenizer.from_pretrained(tmp_path / "base")
    tokenizer.truncation_side = side
    if specials:  # RoBERTa-style <s> ... </s>
        tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>", special_tokens=[("<s>", 1), ("</s>", 2)]
        )
    calls = []
    original = tokenizer._batch_encode_plus

    def counting(*args, **kwargs):
        rows = args[0] if args else kwargs["batch_text_or_text_pairs"]
        calls.append(len(rows))
        return original(*args, **kwargs)

    tokenizer._batch_encode_plus = counting
    features, fraction = tokenize_once(tokenizer, TEXTS, max_length=8)
    tokenizer._batch_encode_plus = original

    expected = tokenizer(TEXTS, truncation=True, max_length=8)
    assert features["input_ids"] == expected["input_ids"]
    assert features["attention_mask"] == expected["attention_mask"]
    assert all(len(ids) <= 8 for ids in features["input_ids"])
    # Truncation statistics come from the same pass; special tokens count towards the length.
    lengths = [len(i) for i in tokenizer(TEXTS, truncation=False)["input_ids"]]
    assert fraction == sum(n > 8 for n in lengths) / 4 == (0.75 if specials else 0.5)
    # The split is encoded once (not once for statistics and once for truncation);
    # the only other call is a one-row self-check of the truncation rule.
    assert calls == [len(TEXTS), 1]
    if specials:
        assert all(ids[0] == 1 and ids[-1] == 2 for ids in features["input_ids"])
    assert tokenize_once(tokenizer, TEXTS, max_length=512)[1] == 0.0


def test_threshold_tuning_and_lora_targets(tmp_path):
    threshold, f1 = tune_threshold([0, 0, 1, 1, 1], [0.1, 0.45, 0.42, 0.6, 0.9])
    assert threshold == pytest.approx(0.42) and f1 == pytest.approx(6 / 7)
    assert tune_threshold([0, 1], [0.2, 0.8])[0] == pytest.approx(0.5)  # ties stay near 0.5
    model = tiny_base(tmp_path / "base")
    qv = classifier_target_modules(model, "qv")
    qkv = classifier_target_modules(model, "qkv")
    everything = classifier_target_modules(model, "all-linear")
    assert len(qv) == 2 and len(qkv) == 3 and set(qv) < set(qkv) < set(everything)
    assert not any(name.startswith("classifier") for name in everything)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A tiny classifier trained with every optional flag on, plus its splits."""
    from src.bug_detection.experiment_data import create_splits
    from src.bug_detection.train_lora import train

    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("optimized")
    rows = []
    for i in range(20):
        for label in (0, 1):
            rows.append(
                {
                    "source_id": f"{i:06d}",
                    "sample_id": len(rows) + 1,
                    "code": f"def f{i}(x): return x {'+' if label == 0 else '-'} 1",
                    "label": label,
                    "mutation_type": "clean" if label == 0 else "arithmetic_operator",
                }
            )
    pd.DataFrame(rows).to_csv(root / "data.csv", index=False)
    create_splits(root / "data.csv", root / "splits")
    tiny_base(root / "roberta")
    models = root / "models"
    train(
        Namespace(
            split_dir=root / "splits",
            output=models / "codebert_lora",
            base_model=str(root / "roberta"),
            revision="main",
            device="cpu",
            precision="auto",
            epochs=2,
            batch_size=4,
            eval_batch_size=8,
            accumulation=1,
            max_length=32,
            gradient_checkpointing=False,
            class_weights=True,
            lora_targets="qkv",
            lora_r=4,
            threshold_tune=True,
        )
    )
    feedback = root / "user_feedback.jsonl"
    feedback.write_text(
        "".join(
            json.dumps({"code": code, "actual_label": label, "is_mistake": True, "fixed_code": fixed}) + "\n"
            for code, label, fixed in [
                ("def g(x): return x - 1", "BUGGY", "def g(x): return x + 1"),
                ("def h(x): return x + 1", "CLEAN", None),
                ("def broken(:", "BUGGY", None),  # invalid syntax is skipped
            ]
        )
    )
    return {"root": root, "models": models, "splits": root / "splits", "feedback": feedback}


def test_optimized_training_artifacts_stay_loadable(trained):
    from src.inference.models import LoraPredictor

    folder = trained["models"] / "codebert_lora"
    config = json.loads((folder / "training_config.json").read_text())
    report = json.loads((folder / "metrics.json").read_text())
    assert config["status"] == "trained" and (folder / "adapter/adapter_model.safetensors").exists()
    runtime = config["runtime"]
    assert runtime["precision"] == "fp32" and runtime["peak_gpu_memory_bytes"] is None
    assert runtime["wall_clock_seconds"] > 0 and runtime["train_samples_per_second"] > 0
    assert config["hyperparameters"]["lora_targets"] == "qkv"
    assert config["hyperparameters"]["lora_r"] == 4
    assert len(config["hyperparameters"]["class_weights"]) == 2
    assert set(config["truncation_fraction"]) == {"train", "validation", "test"}
    # Same metrics protocol/keys as before; the tuned threshold is stored next to argmax results.
    for split in ("validation", "test", "validation_argmax", "test_argmax"):
        assert {"accuracy", "bug_f1", "classification_report", "confusion_matrix"} <= set(report[split])
    assert 0.05 <= config["decision_threshold"] <= 0.95

    predictor = LoraPredictor(folder)
    result = predictor.predict("def f(x): return x - 1")
    threshold = config["decision_threshold"]
    assert result["label"] == ("BUGGY" if result["bug_probability"] >= threshold else "CLEAN")
    predictor.config.pop("decision_threshold")  # without one, argmax is used as before
    plain = predictor.predict("def f(x): return x - 1")
    assert plain["label"] == ("BUGGY" if plain["bug_probability"] > 0.5 else "CLEAN")


def test_feedback_retrain_accepts_into_versioned_dir(trained):
    from src.inference.models import LoraPredictor

    models, base = trained["models"], trained["models"] / "codebert_lora"
    before = {p: p.stat().st_mtime_ns for p in base.rglob("*") if p.is_file()}
    seen = []
    result = rf.retrain_lora(
        feedback_file=trained["feedback"],
        base_lora_dir=base,
        model_root=models,
        split_dir=trained["splits"],
        epochs=1,
        max_f1_drop=1.0,  # always accept: this test is about the mechanics
        progress=lambda **state: seen.append(state),
    )
    out = models / result["output_dir"].rsplit("/", 1)[-1]
    assert result["status"] == "accepted"
    assert out.parent == models and out.name.startswith("codebert_lora_v2_") and out != base
    # The parent adapter was only read, never written.
    assert before == {p: p.stat().st_mtime_ns for p in base.rglob("*") if p.is_file()}
    assert rf.read_active_dir(models) == out
    config = json.loads((out / "training_config.json").read_text())
    assert config["status"] == "trained" and config["parent_lora_dir"] == str(base)
    assert config["feedback_mix"]["feedback_rows"] == 3  # 2 valid rows + 1 fixed_code
    assert config["feedback_mix"]["replay_rows"] == 12 and config["feedback_mix"]["total_rows"] == 18
    assert config["feedback_gate"]["accepted"] is True
    assert config["feedback_gate"]["old_source"] == "recorded metrics.json"  # no extra validation pass
    assert set(result["new_metrics"]) == set(rf.GATE_KEYS)
    assert json.loads((out / "metrics.json").read_text())["validation"]["bug_f1"] == result["new_metrics"]["bug_f1"]
    assert [s["phase"] for s in seen] == ["preparing", "training", "evaluating_new", "saving"]
    assert seen[1]["epoch"] == 1 and seen[1]["loss"] > 0
    assert 0 <= LoraPredictor(out).predict("def g(x): return x - 1")["bug_probability"] <= 1


def test_feedback_retrain_rejects_worse_model(trained, monkeypatch):
    models = trained["models"]
    active = rf.read_active_dir(models)
    pointer = (models / rf.ACTIVE_POINTER).read_text() if (models / rf.ACTIVE_POINTER).exists() else None
    worse = {"accuracy": 0.1, "bug_precision": 0.0, "bug_recall": 0.0, "bug_f1": 0.0}
    monkeypatch.setattr(rf, "cached_validation_metrics", lambda *_: dict(worse, bug_f1=0.9))
    monkeypatch.setattr(rf, "evaluate_adapter", lambda *a, **k: dict(worse))
    result = rf.retrain_lora(
        feedback_file=trained["feedback"],
        base_lora_dir=active,
        model_root=models,
        split_dir=trained["splits"],
        epochs=1,
    )
    assert result["status"] == "rejected_worse"
    assert result["active_model_dir"] == str(active)
    assert rf.read_active_dir(models) == active  # pointer untouched, no swap
    assert ((models / rf.ACTIVE_POINTER).read_text() if pointer else None) == pointer
    rejected = models / result["output_dir"].rsplit("/", 1)[-1]
    assert rejected != active and rejected.name.startswith("codebert_lora_v")
    gate = json.loads((rejected / "training_config.json").read_text())["feedback_gate"]
    assert gate["accepted"] is False and gate["old"]["bug_f1"] == 0.9 and gate["new"]["bug_f1"] == 0.0


def test_feedback_retrain_refuses_to_overwrite_its_source(trained):
    base = trained["models"] / "codebert_lora"
    with pytest.raises(ValueError, match="must differ"):
        rf.retrain_lora(trained["feedback"], base, output_dir=base, split_dir=trained["splits"], epochs=1)
