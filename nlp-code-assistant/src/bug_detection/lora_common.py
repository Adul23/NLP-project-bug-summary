"""Optional transformer helpers; importing baseline/API modules does not import torch."""

import json
from pathlib import Path


def add_classifier_lora(model):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model

    if model.config.model_type != "roberta":
        raise ValueError("This classifier pipeline supports CodeBERT/RoBERTa encoders")
    targets = [
        name
        for name, layer in model.named_modules()
        if isinstance(layer, torch.nn.Linear)
        and name.endswith((".attention.self.query", ".attention.self.value"))
    ]
    if not targets or not hasattr(model, "classifier"):
        raise ValueError(
            "No supported attention projections or classification head found"
        )
    adapted = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
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


def hardware_options(device):
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu")
    cuda = device != "cpu" and torch.cuda.is_available()
    return {
        "use_cpu": not cuda,
        "bf16": cuda and torch.cuda.is_bf16_supported(),
        "fp16": False,
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
