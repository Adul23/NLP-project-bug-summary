"""Load a CodeT5 summarizer checkpoint: a full model folder or a LoRA adapter folder.

A LoRA folder is recognised by adapter_config.json. The base model named in that
file is loaded from the local cache only (nothing is downloaded), then the adapter
is attached. Used by evaluate_model.py and summarize.py.
"""

from __future__ import annotations

import json
from pathlib import Path

ADAPTER_CONFIG = "adapter_config.json"


def is_lora_checkpoint(path: str | Path) -> bool:
    return (Path(path) / ADAPTER_CONFIG).is_file()


def is_full_checkpoint(path: str | Path) -> bool:
    return (Path(path) / "config.json").is_file() and not is_lora_checkpoint(path)


def artifact_size_mb(path: str | Path) -> float:
    """Size of what must be shipped, in MB (10^6 bytes).

    Full model: every file in the folder. LoRA: only the adapter weights and config
    (the tokenizer and the base model come from the base checkpoint).
    """
    path = Path(path)
    if is_lora_checkpoint(path):
        files = [f for f in path.iterdir() if f.name.startswith("adapter_")]
    else:
        files = [f for f in path.rglob("*") if f.is_file()]
    return round(sum(f.stat().st_size for f in files) / 1e6, 2)


def load_model_and_tokenizer(checkpoint: str | Path, device: str, merge_lora: bool = False):
    """Return (model on device in eval mode, tokenizer, kind) with kind 'full' or 'lora'."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    checkpoint = Path(checkpoint)
    if is_lora_checkpoint(checkpoint):
        from peft import PeftModel

        base_name = json.loads((checkpoint / ADAPTER_CONFIG).read_text())[
            "base_model_name_or_path"
        ]
        base = AutoModelForSeq2SeqLM.from_pretrained(base_name, local_files_only=True)
        model = PeftModel.from_pretrained(base, checkpoint)
        if merge_lora:
            model = model.merge_and_unload()
        has_tokenizer = (checkpoint / "tokenizer_config.json").is_file()
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint if has_tokenizer else base_name, local_files_only=True
        )
        kind = "lora"
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        kind = "full"
    return model.to(device).eval(), tokenizer, kind
