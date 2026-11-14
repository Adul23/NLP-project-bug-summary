"""Self-improving loop: fine-tunes the active CodeBERT LoRA adapter on collected human feedback.

Feedback rows are mixed with a replay sample of the original training split (against
catastrophic forgetting), the result is written to a new versioned folder, and it only
becomes the active model if validation BUGGY F1 does not drop by more than MAX_F1_DROP.
Importing this module does not import torch.
"""

import argparse
import ast
import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path

from src.bug_detection.experiment_data import DEFAULT_SPLITS, load_splits
from src.bug_detection.split_dataset import ROOT
from src.bug_detection.train_tfidf import evaluate_predictions

DEFAULT_NAME = "codebert_lora"
ACTIVE_POINTER = "codebert_lora_active.json"
MAX_F1_DROP = 0.01
GATE_KEYS = ("accuracy", "bug_precision", "bug_recall", "bug_f1")
_VERSION = re.compile(rf"^{DEFAULT_NAME}_v(\d+)(?:_|$)")


def load_feedback_records(feedback_path: Path, mistakes_only: bool = False):
    """Читает user_feedback.jsonl, проверяет AST и убирает дубликаты."""
    feedback_path = Path(feedback_path)
    if not feedback_path.exists():
        raise FileNotFoundError(f"Feedback file not found: {feedback_path}")

    seen_codes = set()
    records = []

    with open(feedback_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if mistakes_only and not row.get("is_mistake", False):
                continue

            code = row["code"].strip()
            # Строгая проверка синтаксиса (как в основном пайплайне)
            try:
                ast.parse(code)
            except SyntaxError:
                continue

            if code not in seen_codes:
                seen_codes.add(code)
                label = 1 if row["actual_label"].upper() == "BUGGY" else 0
                records.append({"code": code, "label": label})

            # Если человек прислал исправленный код (fixed_code), добавляем его как CLEAN (0)
            fixed = row.get("fixed_code")
            if fixed and fixed.strip():
                fixed_clean = fixed.strip()
                try:
                    ast.parse(fixed_clean)
                    if fixed_clean not in seen_codes:
                        seen_codes.add(fixed_clean)
                        records.append({"code": fixed_clean, "label": 0})
                except SyntaxError:
                    pass

    return records


def read_base_config(base_lora_dir):
    """training_config.json of an adapter folder; raises if the artifacts are missing."""
    base_lora_dir = Path(base_lora_dir)
    cfg_path = base_lora_dir / "training_config.json"
    if not cfg_path.exists() or not (base_lora_dir / "adapter").is_dir():
        raise FileNotFoundError(f"Не найдены артефакты LoRA в {base_lora_dir}")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


# --- versioned folders and the active-model pointer ---------------------------------


def read_active_dir(model_root):
    """Folder of the active CodeBERT adapter: the pointer target, else models/codebert_lora."""
    model_root = Path(model_root)
    default = model_root / DEFAULT_NAME
    try:
        pointer = json.loads((model_root / ACTIVE_POINTER).read_text(encoding="utf-8"))
        target = Path(pointer["active_dir"])
    except (OSError, ValueError, KeyError, TypeError):
        return default
    target = target if target.is_absolute() else model_root / target
    return target if (target / "training_config.json").exists() else default


def write_active_pointer(model_root, directory, previous=None):
    """Atomically point models/codebert_lora_active.json at a new adapter folder."""
    model_root, directory = Path(model_root), Path(directory)
    try:
        stored = str(directory.resolve().relative_to(model_root.resolve()))
    except ValueError:
        stored = str(directory.resolve())
    pointer = {
        "active_dir": stored,
        "previous_dir": str(previous) if previous else None,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    temporary = model_root / (ACTIVE_POINTER + ".tmp")
    temporary.write_text(json.dumps(pointer, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, model_root / ACTIVE_POINTER)
    return pointer


def next_version_dir(model_root, now=None):
    """models/codebert_lora_v{N}_{timestamp}; the original codebert_lora counts as v1."""
    model_root = Path(model_root)
    versions = [1]
    if model_root.exists():
        for child in model_root.iterdir():
            match = _VERSION.match(child.name)
            if match and child.is_dir():
                versions.append(int(match.group(1)))
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return model_root / f"{DEFAULT_NAME}_v{max(versions) + 1}_{stamp}"


# --- replay mixing and the swap gate ------------------------------------------------


def build_training_mix(
    feedback, train_frame, replay_ratio=4.0, feedback_oversample=2, seed=42
):
    """Feedback rows (oversampled) plus a label-stratified replay sample of the train split.

    replay_ratio is the number of original rows per unique feedback row. The replay keeps the
    train split's clean/buggy proportions and skips code that also appears in the feedback.
    """
    if replay_ratio < 0 or feedback_oversample < 1:
        raise ValueError("replay ratio must be >= 0 and feedback oversample >= 1")
    if not feedback:
        raise ValueError("Нет валидных записей в файле фидбека для дообучения.")
    feedback_codes = {row["code"] for row in feedback}
    pool = train_frame[~train_frame.code.str.strip().isin(feedback_codes)]
    wanted = min(int(round(replay_ratio * len(feedback))), len(pool))
    replay = []
    if wanted:
        buggy_share = float((pool.label == 1).mean())
        counts = {1: min(int(round(wanted * buggy_share)), int((pool.label == 1).sum()))}
        counts[0] = min(wanted - counts[1], int((pool.label == 0).sum()))
        for label, count in sorted(counts.items()):
            if count:
                sample = pool[pool.label == label].sample(n=count, random_state=seed)
                replay += [
                    {"code": code, "label": label, "source": "replay"}
                    for code in sample.code.tolist()
                ]
    records = [
        {**row, "source": "feedback"} for row in feedback
    ] * feedback_oversample + replay
    summary = {
        "feedback_rows": len(feedback),
        "feedback_oversample": feedback_oversample,
        "replay_ratio": replay_ratio,
        "replay_rows": len(replay),
        "replay_buggy": sum(row["label"] for row in replay),
        "total_rows": len(records),
    }
    return records, summary


def gate_decision(old_f1, new_f1, max_drop=MAX_F1_DROP):
    """Accept the new adapter unless validation BUGGY F1 dropped by more than max_drop."""
    return new_f1 >= old_f1 - max_drop - 1e-12


def gate_frame(validation, samples=None, seed=42):
    """Full validation split, or a label-stratified subsample for faster CPU gating."""
    if not samples or samples >= len(validation):
        return validation
    share = samples / len(validation)
    parts = [
        group.sample(n=max(1, int(round(len(group) * share))), random_state=seed)
        for _, group in validation.groupby("label")
    ]
    import pandas as pd

    return pd.concat(parts).reset_index(drop=True)


def cached_validation_metrics(lora_dir, manifest):
    """Full-validation metrics recorded next to an adapter, if they match this split."""
    try:
        report = json.loads((Path(lora_dir) / "metrics.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    metrics = report.get("validation")
    if (
        report.get("split_manifest") != manifest
        or report.get("validation_subsample")
        or not isinstance(metrics, dict)
        or "bug_f1" not in metrics
    ):
        return None
    return metrics


def _autocast(device):
    import torch
    from src.bug_detection.lora_common import native_bf16

    if device.type != "cuda":
        return torch.autocast("cpu", enabled=False), False
    dtype = torch.bfloat16 if native_bf16() else torch.float16
    return torch.autocast("cuda", dtype=dtype), dtype is torch.float16


def evaluate_adapter(
    model, tokenizer, frame, max_length, device, batch_size=64, threshold=None
):
    """Validation metrics with the shared protocol (argmax, or the stored threshold)."""
    import numpy as np
    import torch
    from src.bug_detection.lora_common import softmax_bug_probability, tokenize_once
    from transformers import DataCollatorWithPadding

    features, _ = tokenize_once(tokenizer, frame.code.tolist(), max_length)
    order = sorted(
        range(len(frame)), key=lambda i: len(features["input_ids"][i]), reverse=True
    )
    collate = DataCollatorWithPadding(tokenizer)
    logits = np.zeros((len(frame), 2), dtype=np.float64)
    was_training = model.training
    model.eval()
    context, _ = _autocast(device)
    with torch.inference_mode(), context:
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            batch = collate(
                [
                    {
                        "input_ids": features["input_ids"][i],
                        "attention_mask": features["attention_mask"][i],
                    }
                    for i in rows
                ]
            ).to(device)
            logits[rows] = model(**batch).logits.float().cpu().numpy()
    model.train(was_training)
    if threshold is None:
        predictions = logits.argmax(-1)
    else:
        predictions = (softmax_bug_probability(logits) >= threshold).astype(int)
    return evaluate_predictions(frame.label.to_numpy(), predictions)


def retrain_lora(
    feedback_file: Path,
    base_lora_dir: Path,
    output_dir: Path | None = None,
    epochs: int = 3,
    lr: float = 5e-5,
    batch_size: int = 8,
    mistakes_only: bool = False,
    *,
    model_root: Path | None = None,
    split_dir: Path = DEFAULT_SPLITS,
    replay_ratio: float = 4.0,
    feedback_oversample: int = 2,
    max_f1_drop: float = MAX_F1_DROP,
    gate_samples: int | None = None,
    update_pointer: bool = True,
    progress=None,
):
    """Retrain on feedback + replay, gate on validation BUGGY F1, return a result dict.

    Result "status" is "accepted" (pointer updated when update_pointer) or "rejected_worse".
    The new adapter is always written to a fresh folder, never to base_lora_dir.
    """
    report = progress or (lambda **_: None)
    started = time.perf_counter()
    base_lora_dir = Path(base_lora_dir)
    model_root = Path(model_root) if model_root else base_lora_dir.parent
    report(phase="preparing")

    feedback = load_feedback_records(feedback_file, mistakes_only=mistakes_only)
    if not feedback:
        raise ValueError("Нет валидных записей в файле фидбека для дообучения.")
    train_cfg = read_base_config(base_lora_dir)
    try:
        (training, validation, _), manifest = load_splits(split_dir)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Shared splits not found in {split_dir}; they are needed for replay and the swap gate"
        ) from exc
    records, mix = build_training_mix(
        feedback, training, replay_ratio, feedback_oversample
    )

    output_dir = Path(output_dir) if output_dir else next_version_dir(model_root)
    if output_dir.resolve() == base_lora_dir.resolve():
        raise ValueError("Output folder must differ from the adapter being retrained")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Use an empty output directory; do not overwrite trained artifacts")

    import torch
    from peft import PeftModel
    from src.bug_detection.lora_common import tokenize_once
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
    )

    base_model_name = (
        train_cfg.get("base_model_name_or_path")
        or train_cfg.get("base_model")
        or "microsoft/codebert-base"
    )
    revision = train_cfg.get("base_model_revision") or train_cfg.get("revision")
    max_length = int(train_cfg.get("max_length", 512))
    threshold = train_cfg.get("decision_threshold")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(42)

    # Загружаем токенизатор и базовую модель из локального кэша
    adapter_dir = base_lora_dir / "adapter"
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, local_files_only=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,
        id2label={0: "CLEAN", 1: "BUGGY"},
        label2id={"CLEAN": 0, "BUGGY": 1},
        revision=revision,
        local_files_only=True,
    )
    # Загружаем наш обученный LoRA-адаптер в режиме дообучения (is_trainable=True)
    model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True)
    model.to(device)

    gate_rows = gate_frame(validation, gate_samples)
    subsample = len(gate_rows) if len(gate_rows) < len(validation) else None
    old_metrics = None if subsample else cached_validation_metrics(base_lora_dir, manifest)
    old_source = "recorded metrics.json"
    if old_metrics is None:
        report(phase="evaluating_current")
        old_metrics = evaluate_adapter(
            model, tokenizer, gate_rows, max_length, device, threshold=threshold
        )
        old_source = "evaluated"

    # Dynamic padding: rows are tokenized once and padded per batch, not to max_length.
    features, _ = tokenize_once(tokenizer, [r["code"] for r in records], max_length)
    dataset = [
        {
            "input_ids": features["input_ids"][i],
            "attention_mask": features["attention_mask"][i],
            "labels": records[i]["label"],
        }
        for i in range(len(records))
    ]
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
        collate_fn=DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8 if cuda else None
        ),
        pin_memory=cuda,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    context, needs_scaler = _autocast(device)
    scaler = torch.amp.GradScaler("cuda", enabled=needs_scaler)

    model.train()
    history = []
    train_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with context:
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item())

        avg_loss = round(total_loss / len(loader), 4)
        history.append({"epoch": epoch, "loss": avg_loss})
        report(phase="training", epoch=epoch, epochs=epochs, loss=avg_loss)
        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss} (samples: {len(records)})")
    train_seconds = time.perf_counter() - train_started

    report(phase="evaluating_new", epoch=epochs, epochs=epochs, loss=history[-1]["loss"])
    new_metrics = evaluate_adapter(
        model, tokenizer, gate_rows, max_length, device, threshold=threshold
    )
    accepted = gate_decision(old_metrics["bug_f1"], new_metrics["bug_f1"], max_f1_drop)
    gate = {
        "metric": "validation bug_f1",
        "old": {key: old_metrics[key] for key in GATE_KEYS},
        "new": {key: new_metrics[key] for key in GATE_KEYS},
        "old_source": old_source,
        "max_drop": max_f1_drop,
        "validation_rows": len(gate_rows),
        "accepted": accepted,
    }

    # Сохраняем обновленный адаптер и конфигурацию (в новую версионированную папку)
    report(phase="saving", epoch=epochs, epochs=epochs, loss=history[-1]["loss"])
    out_adapter = output_dir / "adapter"
    out_adapter.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_adapter)
    tokenizer.save_pretrained(out_adapter)

    updated_cfg = {
        **train_cfg,
        "self_improved_at": datetime.now(UTC).isoformat(),
        "parent_lora_dir": str(base_lora_dir),
        "feedback_samples_used": len(feedback),
        "feedback_mix": mix,
        "feedback_epochs": epochs,
        "feedback_learning_rate": lr,
        "feedback_batch_size": batch_size,
        "feedback_loss_history": history,
        "feedback_gate": gate,
        "feedback_runtime": {
            "wall_clock_seconds": round(time.perf_counter() - started, 2),
            "train_seconds": round(train_seconds, 2),
            "train_samples_per_second": round(
                len(records) * epochs / max(train_seconds, 1e-9), 2
            ),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated())
            if cuda
            else None,
            "device": str(device),
        },
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(updated_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    # Validation only: the test split is never touched by feedback retraining.
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "model": "CodeBERT + LoRA",
                "status": "trained",
                "source": "feedback_retrain",
                "split_manifest": manifest,
                "validation": new_metrics,
                "validation_subsample": subsample,
                "parent_validation": gate["old"],
                "accepted": accepted,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if accepted and update_pointer:
        write_active_pointer(model_root, output_dir, previous=base_lora_dir)

    print(
        f"{'Принято' if accepted else 'Отклонено (хуже на валидации)'}: BUGGY F1 "
        f"{gate['old']['bug_f1']:.4f} -> {gate['new']['bug_f1']:.4f}; сохранено в {output_dir}"
    )
    return {
        "status": "accepted" if accepted else "rejected_worse",
        "output_dir": str(output_dir),
        "active_model_dir": str(output_dir if accepted else base_lora_dir),
        "old_metrics": gate["old"],
        "new_metrics": gate["new"],
        "gate": gate,
        "history": history,
        "samples": mix,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Retrain CodeBERT LoRA on human feedback."
    )
    parser.add_argument(
        "--feedback-file",
        type=Path,
        default=ROOT / "data" / "feedback" / "user_feedback.jsonl",
    )
    parser.add_argument("--model-root", type=Path, default=ROOT / "models")
    parser.add_argument(
        "--base-lora-dir",
        type=Path,
        default=None,
        help="adapter to continue from (default: the active one in --model-root)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="default: a new models/codebert_lora_v{N}_{timestamp} folder",
    )
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--mistakes-only",
        action="store_true",
        help="Учиться только на примерах, где модель ошиблась",
    )
    parser.add_argument(
        "--replay-ratio",
        type=float,
        default=4.0,
        help="original train rows mixed in per feedback row (0 disables replay)",
    )
    parser.add_argument("--feedback-oversample", type=int, default=2)
    parser.add_argument(
        "--max-f1-drop",
        type=float,
        default=MAX_F1_DROP,
        help="largest allowed validation BUGGY F1 drop before the new adapter is rejected",
    )
    parser.add_argument(
        "--gate-samples",
        type=int,
        default=None,
        help="gate on a stratified validation subsample of this size (faster on CPU)",
    )
    args = parser.parse_args()

    result = retrain_lora(
        feedback_file=args.feedback_file,
        base_lora_dir=args.base_lora_dir or read_active_dir(args.model_root),
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        mistakes_only=args.mistakes_only,
        model_root=args.model_root,
        split_dir=args.split_dir,
        replay_ratio=args.replay_ratio,
        feedback_oversample=args.feedback_oversample,
        max_f1_drop=args.max_f1_drop,
        gate_samples=args.gate_samples,
    )
    print(json.dumps({k: result[k] for k in ("status", "output_dir", "gate")}, indent=2))


if __name__ == "__main__":
    main()
