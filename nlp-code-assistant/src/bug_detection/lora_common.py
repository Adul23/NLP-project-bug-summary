"""Optional transformer helpers; importing baseline/API modules does not import torch."""

import json
from pathlib import Path

LORA_TARGETS = {
    "qv": (".attention.self.query", ".attention.self.value"),
    "qkv": (".attention.self.query", ".attention.self.key", ".attention.self.value"),
}


def classifier_target_modules(model, targets="qv"):
    """Names of the linear layers that receive LoRA updates (never the classifier head)."""
    import torch

    if targets not in (*LORA_TARGETS, "all-linear"):
        raise ValueError(f"Unknown LoRA targets: {targets}")
    return [
        name
        for name, layer in model.named_modules()
        if isinstance(layer, torch.nn.Linear)
        and not name.startswith("classifier")
        and (targets == "all-linear" or name.endswith(LORA_TARGETS[targets]))
    ]


def add_classifier_lora(model, targets="qv", r=8):
    from peft import LoraConfig, TaskType, get_peft_model

    if model.config.model_type != "roberta":
        raise ValueError("This classifier pipeline supports CodeBERT/RoBERTa encoders")
    if r < 1:
        raise ValueError("LoRA rank must be positive")
    targets = classifier_target_modules(model, targets)
    if not targets or not hasattr(model, "classifier"):
        raise ValueError(
            "No supported attention projections or classification head found"
        )
    adapted = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=r,
            lora_alpha=2 * r,  # 16 for the default r=8
            lora_dropout=0.1,
            target_modules=targets,
            modules_to_save=["classifier"],
        ),
    )
    if not any(
        p.requires_grad for n, p in adapted.named_parameters() if "classifier" in n
    ):
        raise ValueError("Classification head is not trainable")
    return adapted, targets


def add_summarizer_lora(model):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model

    if model.config.model_type != "t5":
        raise ValueError("This summarizer pipeline supports T5/CodeT5")
    targets = [
        name
        for name, layer in model.named_modules()
        if isinstance(layer, torch.nn.Linear) and name.endswith((".q", ".v"))
    ]
    if not targets:
        raise ValueError("No T5 attention q/v projections found")
    return get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=targets,
        ),
    ), targets


PRECISIONS = ("auto", "fp32", "fp16", "bf16")


def select_precision(requested, cuda, bf16_supported, allow_fp16_auto=True):
    """Pure selection logic: returns "fp32", "fp16" or "bf16".

    auto: bf16 on hardware with native support, otherwise fp16 on CUDA, fp32 on CPU.
    allow_fp16_auto=False keeps auto at fp32 when bf16 is missing (T5 can overflow in fp16).
    """
    if requested not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    if not cuda:
        if requested in ("fp16", "bf16"):
            raise ValueError(f"--precision {requested} requires CUDA; use fp32 or auto")
        return "fp32"
    if requested == "auto":
        if bf16_supported:
            return "bf16"
        return "fp16" if allow_fp16_auto else "fp32"
    if requested == "bf16" and not bf16_supported:
        raise ValueError("This GPU has no native bf16 support; use --precision fp16")
    return requested


def native_bf16():
    """True only for real bf16 hardware; torch>=2.5 otherwise also reports slow emulation (e.g. T4)."""
    import torch

    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:  # older torch without the keyword
        return bool(torch.cuda.is_bf16_supported())


def hardware_options(device, precision="auto", allow_fp16_auto=True):
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu")
    cuda = device != "cpu" and torch.cuda.is_available()
    chosen = select_precision(precision, cuda, cuda and native_bf16(), allow_fp16_auto)
    if cuda:
        # TF32 speeds up fp32 matmul/conv on Ampere and newer; a no-op on older GPUs.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return {"use_cpu": not cuda, "bf16": chosen == "bf16", "fp16": chosen == "fp16"}


def precision_name(options):
    return "bf16" if options["bf16"] else "fp16" if options["fp16"] else "fp32"


def loader_options(options):
    """Dataloader settings: worker processes and pinned memory only help on CUDA."""
    cuda = not options["use_cpu"]
    return {
        "dataloader_num_workers": 2 if cuda else 0,
        "dataloader_pin_memory": cuda,
    }


def pad_multiple(options):
    """Pad to a multiple of 8 under mixed precision (tensor-core friendly shapes)."""
    return 8 if options["bf16"] or options["fp16"] else None


def tokenize_once(tokenizer, texts, max_length):
    """Tokenize each text once; return (features, truncation_fraction).

    The untruncated lengths give the truncation fraction, and over-long rows are cut
    in the same pass exactly as the tokenizer would: content is trimmed, leading and
    trailing special tokens are kept.
    """
    if not texts:
        raise ValueError("Nothing to tokenize")
    full = tokenizer(
        texts,
        truncation=False,
        verbose=False,
        return_special_tokens_mask=True,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    left = getattr(tokenizer, "truncation_side", "right") == "left"
    input_ids, checked, long_rows = [], False, 0
    for text, ids, special in zip(
        texts, full["input_ids"], full["special_tokens_mask"]
    ):
        if len(ids) > max_length:
            long_rows += 1
            head = next((i for i, flag in enumerate(special) if not flag), len(ids))
            tail = next((i for i, flag in enumerate(reversed(special)) if not flag), 0)
            keep = max_length - head - tail
            body = ids[head : len(ids) - tail]
            body = body[len(body) - keep :] if left else body[:keep]
            ids = ids[:head] + body + ids[len(ids) - tail :]
            if (
                not checked
            ):  # one cheap self-check against the tokenizer's own truncation
                checked = True
                expected = tokenizer(
                    text, truncation=True, max_length=max_length, verbose=False
                )["input_ids"]
                if ids != expected:
                    return _tokenize_twice(tokenizer, texts, max_length, full)
        input_ids.append(ids)
    return (
        {"input_ids": input_ids, "attention_mask": [[1] * len(i) for i in input_ids]},
        long_rows / len(texts),
    )


def _tokenize_twice(tokenizer, texts, max_length, full):
    """Fallback for tokenizers whose truncation is not a plain content trim."""
    fraction = sum(len(i) > max_length for i in full["input_ids"]) / len(texts)
    cut = tokenizer(
        texts, truncation=True, max_length=max_length, return_token_type_ids=False
    )
    return {
        "input_ids": cut["input_ids"],
        "attention_mask": cut["attention_mask"],
    }, fraction


def class_weights(labels, classes=2):
    """Balanced weights n / (classes * count), e.g. 7000/3320 -> [0.737, 1.554]."""
    labels = list(labels)
    counts = [labels.count(c) for c in range(classes)]
    if min(counts) == 0:
        raise ValueError("Every class needs at least one training row")
    return [len(labels) / (classes * count) for count in counts]


def weighted_trainer(weights):
    """Trainer subclass using class-weighted cross-entropy."""
    import torch
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss = torch.nn.functional.cross_entropy(
                logits.float(),
                labels,
                weight=torch.tensor(weights, dtype=torch.float, device=logits.device),
            )
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


def tune_threshold(labels, bug_probabilities):
    """Validation threshold maximizing BUGGY F1; ties prefer the value closest to 0.5."""
    import numpy as np

    labels = np.asarray(labels)
    probabilities = np.asarray(bug_probabilities, dtype=float)
    best = (-1.0, 0.0, 0.5)
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        predicted = probabilities >= threshold
        tp = int(np.sum(predicted & (labels == 1)))
        fp = int(np.sum(predicted & (labels == 0)))
        fn = int(np.sum(~predicted & (labels == 1)))
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        candidate = (f1, -abs(float(threshold) - 0.5), float(threshold))
        best = max(best, candidate)
    return best[2], best[0]


def softmax_bug_probability(logits):
    import numpy as np

    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp[:, 1] / exp.sum(axis=-1)


def runtime_report(train_metrics, started, options, samples):
    """Wall-clock, throughput and peak GPU memory for training_config.json."""
    import time

    import torch

    cuda = not options["use_cpu"]
    return {
        "wall_clock_seconds": round(time.perf_counter() - started, 2),
        "train_seconds": train_metrics.get("train_runtime"),
        "train_samples_per_second": train_metrics.get("train_samples_per_second"),
        "train_steps_per_second": train_metrics.get("train_steps_per_second"),
        "train_samples": samples,
        "precision": precision_name(options),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated())
        if cuda
        else None,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
    }


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def require_empty(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Use an empty output directory; do not overwrite trained artifacts"
        )
    output.mkdir(parents=True, exist_ok=True)
    return output
