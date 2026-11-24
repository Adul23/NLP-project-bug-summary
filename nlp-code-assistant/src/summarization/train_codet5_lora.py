"""LoRA fine-tuning of CodeT5-small for one-sentence code summaries (MPS/CUDA/CPU).

Same data, tokenization, batch size, seed and early stopping as train_codet5.py (full
fine-tuning), so the two are comparable. Only the optimizer targets differ: the base
model is frozen and small low-rank adapters are trained. Weights stay fp32 (T5 overflows
in fp16). Only train and validation splits are read; test is never touched here.

Outputs (default: checkpoints/codet5_summarizer_lora, outside git; folder must be empty):
  best/                  best adapter by validation loss (adapter_config.json, adapter weights)
  train_log.jsonl        one JSON line per logged step and per epoch
  training_summary.json  settings, trainable parameters, adapter size, timing, history

Quick smoke test with timing:
  python -m src.summarization.train_codet5_lora --limit 64 --epochs 1 --log-every 1
"""

from __future__ import annotations

from pathlib import Path

from src.summarization import train_codet5 as full
from src.summarization.model_loading import artifact_size_mb

DEFAULT_OUTPUT = full.ROOT / "checkpoints/codet5_summarizer_lora"
DEFAULT_LR = 2e-4
LORA_TARGETS = {
    "qv": ["q", "v"],
    "qkvo": ["q", "k", "v", "o"],
    "all-linear": "all-linear",
}


def build_lora_config(r: int = 8, alpha: int = 16, dropout: float = 0.1, targets: str = "qv"):
    from peft import LoraConfig, TaskType

    if targets not in LORA_TARGETS:
        raise ValueError(f"--lora-targets must be one of {sorted(LORA_TARGETS)}")
    if r < 1:
        raise ValueError("LoRA rank must be at least 1")
    return LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=LORA_TARGETS[targets],
    )


def count_parameters(model) -> tuple[int, int]:
    """(trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, sum(p.numel() for p in model.parameters())


def describe(model, output: Path) -> dict:
    trainable, total = count_parameters(model)
    best = output / "best"
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
        "adapter_size_mb": artifact_size_mb(best) if best.is_dir() else None,
    }


def make_loader(config):
    def load(device):
        from peft import get_peft_model
        from transformers import AutoModelForSeq2SeqLM

        base = AutoModelForSeq2SeqLM.from_pretrained(full.BASE_MODEL)
        return get_peft_model(base, config).to(device)  # fp32

    return load


def main() -> None:
    parser = full.build_parser(
        __doc__.splitlines()[0],
        DEFAULT_LR,
        "adapter folder, must be empty (default: checkpoints/codet5_summarizer_lora; "
        "with --limit: a new checkpoints/codet5_lora_smoke_<time>)",
    )
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--lora-targets", choices=sorted(LORA_TARGETS), default="qv")
    args = parser.parse_args()
    config = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_targets)
    full.train(
        args,
        load_model=make_loader(config),
        default_output=DEFAULT_OUTPUT,
        smoke_prefix="codet5_lora_smoke",
        describe=describe,
        extra_settings={
            "method": "lora",
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_targets": args.lora_targets,
        },
    )


if __name__ == "__main__":
    main()
