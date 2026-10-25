"""Small local FastAPI backend; submitted code is text and is never executed."""

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from src.bug_detection.split_dataset import ROOT
from src.inference.models import LoraPredictor, TfidfPredictor
from src.summarization.prepare import verified_change
from starlette.responses import JSONResponse

MAX_BODY_BYTES = 128_000
UNCERTAINTY_LOW = 0.40
UNCERTAINTY_HIGH = 0.60


class BodyLimit:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        # Count actual streamed bytes, not just an untrusted Content-Length header.
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > MAX_BODY_BYTES:
                return await JSONResponse(
                    {"detail": "Request exceeds 128000 bytes"}, status_code=413
                )(scope, receive, send)
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {
                    "type": "http.request",
                    "body": bytes(body),
                    "more_body": False,
                }
            return await receive()

        await self.app(scope, replay, send)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    code: str = Field(min_length=1, max_length=20_000)
    model: Literal["tfidf", "codebert"] = "tfidf"

    @field_validator("code")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Code must not be blank")
        return value


class AnalysisRequest(PredictionRequest):
    original_code: str | None = Field(default=None, min_length=1, max_length=20_000)
    explanation: Literal["deterministic", "lora"] = "deterministic"


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    code: str = Field(min_length=1, max_length=20_000)
    model: Literal["tfidf", "codebert"] = "codebert"
    predicted_label: Literal["CLEAN", "BUGGY", "clean", "buggy"]
    actual_label: Literal["CLEAN", "BUGGY", "clean", "buggy"]
    user_explanation: str | None = Field(default=None, min_length=1, max_length=5_000)
    fixed_code: str | None = Field(default=None, min_length=1, max_length=20_000)

    @field_validator("code")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Code must not be blank")
        return value


def create_app(model_root=None, feedback_dir=None):
    root = Path(model_root or os.environ.get("BUG_MODEL_ROOT", ROOT / "models"))
    fb_root = Path(
        feedback_dir or os.environ.get("BUG_FEEDBACK_DIR", ROOT / "data" / "feedback")
    )
    uncertain_file = fb_root / "uncertain_cases.jsonl"
    user_feedback_file = fb_root / "user_feedback.jsonl"

    directories = {
        "tfidf": root / "tfidf_v2",
        "codebert": root / "codebert_lora",
        "summarizer": root / "codet5_lora",
    }
    cache, failures = {}, {}
    lock = threading.Lock()
    fb_lock = threading.Lock()
    api = FastAPI(title="Python Bug Detection and Explanation")
    api.add_middleware(BodyLimit)

    def append_jsonl(path: Path, record: dict):
        with fb_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_if_uncertain(code: str, model: str, prediction: dict):
        prob = prediction.get("bug_probability")
        if (
            isinstance(prob, (int, float))
            and UNCERTAINTY_LOW <= prob <= UNCERTAINTY_HIGH
        ):
            append_jsonl(
                uncertain_file,
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "code": code,
                    "model": model,
                    "predicted_label": prediction.get("label"),
                    "bug_probability": round(float(prob), 4),
                    "reason": "uncertainty_sampling",
                },
            )

    def load(name):
        with lock:
            if name not in cache:
                try:
                    cache[name] = (
                        TfidfPredictor(directories[name])
                        if name == "tfidf"
                        else LoraPredictor(
                            directories[name], summarizer=name == "summarizer"
                        )
                    )
                    failures.pop(name, None)
                except (
                    OSError,
                    ValueError,
                    ImportError,
                    RuntimeError,
                    KeyError,
                    EOFError,
                ) as exc:
                    failures[name] = type(exc).__name__
                    raise HTTPException(
                        503,
                        f"{name} unavailable: train its artifacts and install"
                        " its optional dependencies; see README",
                    ) from exc
            return cache[name]

    @api.get("/health")
    def health():
        return {
            "status": "ok",
            "models": {
                name: {
                    "status": (
                        "loaded"
                        if name in cache
                        else (
                            "load_failed"
                            if name in failures
                            else (
                                "artifacts_present_not_loaded"
                                if (
                                    (folder / "logistic_regression.joblib").exists()
                                    and (folder / "tfidf_vectorizer.joblib").exists()
                                    if name == "tfidf"
                                    else (folder / "training_config.json").exists()
                                    and (
                                        folder / "adapter/adapter_config.json"
                                    ).exists()
                                )
                                else "unavailable"
                            )
                        )
                    ),
                }
                for name, folder in directories.items()
            },
        }

    @api.post("/predict")
    def predict(request: PredictionRequest):
        prediction = load(request.model).predict(request.code)
        log_if_uncertain(request.code, request.model, prediction)
        return prediction

    @api.post("/analyze")
    def analyze(request: AnalysisRequest):
        change = None
        if request.original_code is not None:
            try:
                change = verified_change(request.original_code, request.code)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
        prediction = load(request.model).predict(request.code)
        log_if_uncertain(request.code, request.model, prediction)
        result = {
            **prediction,
            "is_buggy": prediction["label"] == "BUGGY",
            "bug_type": None,
            "explanation": None,
            "suggested_fix": None,
            "explanation_source": "unavailable",
            "verified": False,
        }
        if change:
            result.update(
                bug_type=change["bug_type"],
                explanation=change["explanation"],
                explanation_source="verified_code_pair",
                verified=True,
            )
        elif request.explanation == "lora":
            result.update(load("summarizer").generate(request.code))
        return result

    @api.post("/feedback")
    def feedback(request: FeedbackRequest):
        pred_norm = request.predicted_label.upper()
        actual_norm = request.actual_label.upper()
        is_mistake = pred_norm != actual_norm
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "code": request.code,
            "model": request.model,
            "predicted_label": pred_norm,
            "actual_label": actual_norm,
            "is_mistake": is_mistake,
            "user_explanation": request.user_explanation,
            "fixed_code": request.fixed_code,
        }
        append_jsonl(user_feedback_file, record)
        return {"status": "saved", "is_mistake": is_mistake}

    return api


app = create_app()
