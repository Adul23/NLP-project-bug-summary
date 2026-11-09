"""Self-Improving loop: fine-tunes the existing CodeBERT LoRA adapter on collected human feedback."""

import argparse
import ast
import json
from datetime import UTC, datetime
from pathlib import Path

import torch
from src.bug_detection.split_dataset import ROOT
from torch.utils.data import DataLoader, Dataset


class FeedbackDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=512):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        encoded = self.tokenizer(
            item["code"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.long),
        }


def load_feedback_records(feedback_path: Path, mistakes_only: bool = False):
    """Читает user_feedback.jsonl, проверяет AST и убирает дубликаты."""
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


def retrain_lora(
    feedback_file: Path,
    base_lora_dir: Path,
    output_dir: Path,
    epochs: int = 3,
    lr: float = 5e-5,
    batch_size: int = 4,
    mistakes_only: bool = False,
):
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    records = load_feedback_records(feedback_file, mistakes_only=mistakes_only)
    if not records:
        raise ValueError("Нет валидных записей в файле фидбека для дообучения.")

    cfg_path = base_lora_dir / "training_config.json"
    adapter_dir = base_lora_dir / "adapter"
    if not cfg_path.exists() or not adapter_dir.exists():
        raise FileNotFoundError(f"Не найдены артефакты LoRA в {base_lora_dir}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        train_cfg = json.load(f)

    base_model_name = (
        train_cfg.get("base_model_name_or_path")
        or train_cfg.get("base_model")
        or "microsoft/codebert-base"
    )
    revision = train_cfg.get("base_model_revision") or train_cfg.get("revision")
    max_length = int(train_cfg.get("max_length", 512))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загружаем токенизатор и базовую модель из локального кэша
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, local_files_only=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,
        revision=revision,
        local_files_only=True,
    )

    # Загружаем наш обученный LoRA-адаптер в режиме дообучения (is_trainable=True)
    model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True)
    model.to(device)
    model.train()

    dataset = FeedbackDataset(records, tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    history = []
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        avg_loss = round(total_loss / len(loader), 4)
        history.append({"epoch": epoch, "loss": avg_loss})
        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss} (samples: {len(records)})")

    # Сохраняем обновленный адаптер и конфигурацию
    out_adapter = output_dir / "adapter"
    out_adapter.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_adapter)
    tokenizer.save_pretrained(out_adapter)

    updated_cfg = {
        **train_cfg,
        "self_improved_at": datetime.now(UTC).isoformat(),
        "parent_lora_dir": str(base_lora_dir),
        "feedback_samples_used": len(records),
        "feedback_epochs": epochs,
        "feedback_learning_rate": lr,
        "feedback_loss_history": history,
    }
    with open(output_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(updated_cfg, f, indent=2, ensure_ascii=False)

    print(f"Успешно сохранено в {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Retrain CodeBERT LoRA on human feedback."
    )
    parser.add_argument(
        "--feedback-file",
        type=Path,
        default=ROOT / "data" / "feedback" / "user_feedback.jsonl",
    )
    parser.add_argument(
        "--base-lora-dir",
        type=Path,
        default=ROOT / "models" / "codebert_lora",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "models" / "codebert_lora_v2",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--mistakes-only",
        action="store_true",
        help="Учиться только на примерах, где модель ошиблась",
    )
    args = parser.parse_args()

    retrain_lora(
        feedback_file=args.feedback_file,
        base_lora_dir=args.base_lora_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        mistakes_only=args.mistakes_only,
    )


if __name__ == "__main__":
    main()
