"""One-sentence summary of a Python function with a fine-tuned CodeT5 checkpoint.

    from src.summarization.summarize import summarize
    summarize("def add(a, b):\\n    return a + b")

Two backends: `full` (fully fine-tuned model, checkpoints/codet5_summarizer/best) and
`lora` (adapter on codet5-small, checkpoints/codet5_summarizer_lora/best). The checkpoint
is loaded lazily on the first call. Which folder is used:
  1. the `checkpoint` argument (full model or LoRA adapter, detected by adapter_config.json),
  2. else the CODET5_SUMMARIZER_CHECKPOINT environment variable (same detection),
  3. else the `backend` argument ('full' or 'lora'),
  4. else automatic: the full checkpoint if it exists, otherwise the LoRA one.
Only local files are used (nothing is downloaded) and the code is only tokenized,
never executed.

CLI:  python -m src.summarization.summarize --file my_function.py [--backend lora]
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints/codet5_summarizer/best"
DEFAULT_LORA_CHECKPOINT = PROJECT_ROOT / "checkpoints/codet5_summarizer_lora/best"
BACKENDS = ("auto", "full", "lora")
CHECKPOINT_ENV = "CODET5_SUMMARIZER_CHECKPOINT"
MAX_CODE_CHARS = 20000

# Generation settings chosen on validation (rule fixed before the runs).
NUM_BEAMS = 4
LENGTH_PENALTY = 2.0
MAX_LENGTH = 384  # input tokens, same as training
MAX_NEW_TOKENS = 48  # same as max_target in training


def resolve_checkpoint(checkpoint: str | Path | None = None, backend: str | None = None) -> Path:
    """Pick the checkpoint folder (see the module docstring for the order)."""
    if backend not in (None, *BACKENDS):
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if checkpoint:
        return Path(checkpoint)
    from_env = os.environ.get(CHECKPOINT_ENV)
    if from_env:
        return Path(from_env)
    if backend == "full":
        return DEFAULT_CHECKPOINT
    if backend == "lora":
        return DEFAULT_LORA_CHECKPOINT
    if (DEFAULT_CHECKPOINT / "config.json").is_file():
        return DEFAULT_CHECKPOINT
    if (DEFAULT_LORA_CHECKPOINT / "adapter_config.json").is_file():
        return DEFAULT_LORA_CHECKPOINT
    return DEFAULT_CHECKPOINT  # nothing found: the loader explains what is missing


class _HfBackend:
    """Loads the checkpoint with transformers and generates one summary per call."""

    def __init__(self, checkpoint: Path, device: str | None = None):
        from src.summarization.model_loading import (
            is_full_checkpoint,
            is_lora_checkpoint,
            load_model_and_tokenizer,
        )

        if not (is_full_checkpoint(checkpoint) or is_lora_checkpoint(checkpoint)):
            raise FileNotFoundError(
                f"No CodeT5 checkpoint in {checkpoint}. Train one "
                "(train_codet5 for 'full', train_codet5_lora for 'lora'), or set "
                f"{CHECKPOINT_ENV} to a folder with config.json (full model) or "
                "adapter_config.json (LoRA adapter)."
            )
        import torch

        self._torch = torch
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        # A LoRA adapter is merged into the base weights: same outputs, no adapter overhead.
        self.model, self.tokenizer, self.kind = load_model_and_tokenizer(
            checkpoint, device, merge_lora=True
        )

    def generate(self, code, *, max_length, max_new_tokens, num_beams, length_penalty) -> str:
        encoded = self.tokenizer(
            code, max_length=max_length, truncation=True, return_tensors="pt"
        ).to(self.device)
        with self._torch.inference_mode():
            ids = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=True,
            )
        return self.tokenizer.decode(ids[0], skip_special_tokens=True)


_backend_factory = _HfBackend  # replaced by a fake in tests


def _clean(text: str) -> str:
    """Single line, collapsed whitespace."""
    return " ".join(text.split())


class CodeSummarizer:
    """Lazy wrapper: nothing is loaded until the first summarize() call."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        backend: str | None = None,
        device: str | None = None,
        num_beams: int = NUM_BEAMS,
        length_penalty: float = LENGTH_PENALTY,
        max_length: int = MAX_LENGTH,
        max_new_tokens: int = MAX_NEW_TOKENS,
        backend_factory=None,
    ):
        self.checkpoint = resolve_checkpoint(checkpoint, backend)
        self.device = device
        self.settings = {
            "num_beams": num_beams,
            "length_penalty": length_penalty,
            "max_length": max_length,
            "max_new_tokens": max_new_tokens,
        }
        self._factory = backend_factory
        self._backend = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._backend is not None

    def _get_backend(self):
        with self._lock:
            if self._backend is None:
                factory = self._factory or _backend_factory
                self._backend = factory(self.checkpoint, self.device)
            return self._backend

    def summarize(self, code: str) -> str:
        if not isinstance(code, str):
            raise ValueError("code must be a string")
        if not code.strip():
            raise ValueError("code is empty")
        if len(code) > MAX_CODE_CHARS:
            raise ValueError(f"code is longer than {MAX_CODE_CHARS} characters")
        return _clean(self._get_backend().generate(code, **self.settings))


_SUMMARIZERS: dict[Path, CodeSummarizer] = {}
_SUMMARIZERS_LOCK = threading.Lock()


def summarize(
    code: str, checkpoint: str | Path | None = None, backend: str | None = None
) -> str:
    """Summarize one function; the model for each checkpoint folder is loaded once."""
    path = resolve_checkpoint(checkpoint, backend)
    with _SUMMARIZERS_LOCK:
        if path not in _SUMMARIZERS:
            _SUMMARIZERS[path] = CodeSummarizer(path)
        summarizer = _SUMMARIZERS[path]
    return summarizer.summarize(code)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", type=Path, help="UTF-8 file with one function; default: stdin")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="full model or LoRA adapter folder (overrides --backend)")
    parser.add_argument("--backend", choices=BACKENDS, default="auto",
                        help="full | lora | auto (full if its checkpoint exists, else lora)")
    args = parser.parse_args()
    code = args.file.read_text(encoding="utf-8") if args.file else sys.stdin.read()
    try:
        print(summarize(code, args.checkpoint, args.backend))
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(1, f"Summarizer unavailable or invalid input: {exc}\n")


if __name__ == "__main__":
    main()
