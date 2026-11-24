"""Local artifact loading. Transformer downloads are never triggered by inference."""

import json
from pathlib import Path

import joblib


class TfidfPredictor:
    def __init__(self, folder):
        folder = Path(folder)
        self.vectorizer = joblib.load(folder / "tfidf_vectorizer.joblib")
        self.model = joblib.load(folder / "logistic_regression.joblib")
        if len(self.vectorizer.vocabulary_) != self.model.n_features_in_:
            raise ValueError(
                "Vectorizer and classifier feature dimensions do not match"
            )
        if list(self.model.classes_) != [0, 1]:
            raise ValueError("Artifact label mapping must be 0=CLEAN, 1=BUGGY")

    def predict(self, code):
        features = self.vectorizer.transform([code])
        probability = float(self.model.predict_proba(features)[0, 1])
        label = int(self.model.predict(features)[0])
        return {
            "label": "BUGGY" if label else "CLEAN",
            "bug_probability": probability,
            "confidence": probability if label else 1 - probability,
            "model": "tfidf",
        }


class LoraPredictor:
    def __init__(self, folder, *, summarizer=False, device="cpu", merge=True):
        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        folder = Path(folder)
        config = json.loads((folder / "training_config.json").read_text())
        if config.get("status") != "trained":
            raise ValueError("Adapter has not completed training")
        self.config = config
        self.summarizer = summarizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            folder / "adapter", local_files_only=True
        )
        cls = (
            AutoModelForSeq2SeqLM if summarizer else AutoModelForSequenceClassification
        )
        kwargs = (
            {}
            if summarizer
            else {
                "num_labels": 2,
                "id2label": {0: "CLEAN", 1: "BUGGY"},
                "label2id": {"CLEAN": 0, "BUGGY": 1},
                "attn_implementation": "sdpa",
            }
        )
        base = cls.from_pretrained(
            config["base_model"],
            revision=config["revision"],
            local_files_only=True,
            **kwargs,
        )
        model = PeftModel.from_pretrained(
            base, folder / "adapter", local_files_only=True
        )
        if merge:
            model = (
                model.merge_and_unload()
            )  # LoRA вливается в веса, обученная голова classifier остаётся
        self.model = model.to(device).eval()
        self.torch = torch

    def predict(self, code):
        if self.summarizer:
            raise ValueError("Use generate for a summarizer")
        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=self.config["max_length"],
            return_token_type_ids=False,
        ).to(self.model.device)
        with self.torch.inference_mode():
            probabilities = self.model(**inputs).logits.softmax(-1)[0]
        p = float(probabilities[1])
        # A threshold tuned on validation (train_lora --threshold-tune) replaces argmax if stored.
        threshold = self.config.get("decision_threshold")
        label = (
            int(probabilities.argmax()) if threshold is None else int(p >= threshold)
        )
        return {
            "label": "BUGGY" if label else "CLEAN",
            "bug_probability": p,
            "confidence": p if label else 1 - p,
            "model": "codebert",
        }

    def generate(self, code):
        if not self.summarizer:
            raise ValueError("Not a summarizer")
        from src.summarization.train_lora import PROMPT

        inputs = self.tokenizer(
            PROMPT + code,
            return_tensors="pt",
            truncation=True,
            max_length=self.config["max_length"],
            return_token_type_ids=False,
        ).to(self.model.device)
        with self.torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        return {
            "explanation": self.tokenizer.decode(ids[0], skip_special_tokens=True),
            "explanation_source": "lora_generated_unverified",
            "verified": False,
            "bug_type": None,
            "suggested_fix": None,
        }


class CodeSummarizer:
    """Fully fine-tuned CodeT5-small (src/summarization/train_codet5.py).

    Produces a one-sentence, docstring-style summary of what a function does. This is a
    separate model from the CodeT5 LoRA explainer (LoraPredictor with summarizer=True).
    Input is the raw code with no prompt prefix, exactly as in training.
    """

    DEFAULT_MAX_LENGTH = 384
    DEFAULT_MAX_TARGET = 48

    def __init__(self, folder, *, device="cpu"):
        folder = Path(folder)
        best = folder / "best"
        # Fail before importing torch/transformers when nothing has been trained.
        for name in ("config.json", "model.safetensors"):
            if not (best / name).exists():
                raise FileNotFoundError(f"Missing {best / name}")
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        summary_path = folder / "training_summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        self.max_length = int(summary.get("max_length", self.DEFAULT_MAX_LENGTH))
        self.max_target = int(summary.get("max_target", self.DEFAULT_MAX_TARGET))
        self.tokenizer = AutoTokenizer.from_pretrained(best, local_files_only=True)
        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(best, local_files_only=True)
            .to(device)
            .eval()
        )
        self.torch = torch

    def summarize(self, code):
        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.model.device)
        with self.torch.inference_mode():
            ids = self.model.generate(
                **inputs,
                num_beams=4,
                max_new_tokens=self.max_target,
                early_stopping=True,
                do_sample=False,
            )
        return {
            "summary": self.tokenizer.decode(ids[0], skip_special_tokens=True).strip(),
            "summary_source": "codet5_finetuned",
            "model": "codet5_summarizer",
        }
