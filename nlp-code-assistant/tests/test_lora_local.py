"""Offline random-model integration tests, not pretrained model benchmarks."""

import json
from argparse import Namespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
from src.bug_detection.lora_common import add_classifier_lora, add_summarizer_lora
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    PreTrainedTokenizerFast,
    RobertaConfig,
    RobertaForSequenceClassification,
    T5Config,
    T5ForConditionalGeneration,
)


def tiny_base(folder, summarizer=False):
    folder.mkdir()
    tokenizer = Tokenizer(
        WordLevel(
            {
                "<pad>": 0,
                "<s>": 1,
                "</s>": 2,
                "<unk>": 3,
                "def": 4,
                "return": 5,
                "+": 6,
                "-": 7,
                "x": 8,
                "1": 9,
            },
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        model_max_length=512,
    )
    fast.model_input_names = ["input_ids", "attention_mask"]
    fast.save_pretrained(folder)
    if summarizer:
        model = T5ForConditionalGeneration(
            T5Config(
                vocab_size=10,
                d_model=16,
                d_kv=8,
                d_ff=32,
                num_layers=1,
                num_decoder_layers=1,
                num_heads=2,
                decoder_start_token_id=0,
                eos_token_id=2,
                pad_token_id=0,
            )
        )
    else:
        model = RobertaForSequenceClassification(
            RobertaConfig(
                vocab_size=10,
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=32,
                max_position_embeddings=514,
                pad_token_id=0,
                num_labels=2,
            )
        )
    model.save_pretrained(folder)
    return model


@pytest.mark.parametrize("summarizer", [False, True])
def test_gradients_save_reload(tmp_path, summarizer):
    from peft import PeftModel

    torch.set_num_threads(1)
    base_dir = tmp_path / "base"
    base = tiny_base(base_dir, summarizer)
    model, targets = (add_summarizer_lora if summarizer else add_classifier_lora)(base)
    assert targets
    assert all(
        not p.requires_grad
        for n, p in model.named_parameters()
        if "lora_" not in n and "modules_to_save" not in n
    )
    inputs = {
        "input_ids": torch.tensor([[1, 4, 5, 6, 2]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
        "labels": torch.tensor([[4, 5, 2]]) if summarizer else torch.tensor([1]),
    }
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.01
    )
    loss = model(**inputs).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in model.named_parameters()
        if "lora_" in n
    )
    if not summarizer:
        assert any(
            p.grad is not None
            for n, p in model.named_parameters()
            if "classifier.modules_to_save" in n
        )
    optimizer.step()
    model.eval()
    with torch.no_grad():
        expected = model(**inputs).logits
    model.save_pretrained(tmp_path / "adapter")
    cls = T5ForConditionalGeneration if summarizer else RobertaForSequenceClassification
    loaded = PeftModel.from_pretrained(
        cls.from_pretrained(base_dir), tmp_path / "adapter"
    ).eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded(**inputs).logits, expected)


def test_training_entrypoints(tmp_path):
    import pandas as pd
    from src.bug_detection.experiment_data import create_splits
    from src.bug_detection.train_lora import train as classify
    from src.inference.models import LoraPredictor
    from src.summarization.prepare import prepare
    from src.summarization.train_lora import train as summarize

    torch.set_num_threads(1)
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
    dataset = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(dataset, index=False)
    splits = tmp_path / "splits"
    create_splits(dataset, splits)
    base_dir = tmp_path / "roberta"
    tiny_base(base_dir)
    options = dict(
        split_dir=splits,
        output=tmp_path / "classifier",
        base_model=str(base_dir),
        revision="main",
        device="cpu",
        epochs=1,
        batch_size=2,
        accumulation=1,
        max_length=32,
        gradient_checkpointing=True,
    )
    classify(Namespace(**options))
    result = LoraPredictor(options["output"]).predict("def f(x): return x - 1")
    assert 0 <= result["bug_probability"] <= 1
    assert (options["output"] / "adapter/adapter_model.safetensors").exists()
    t5_dir = tmp_path / "t5"
    tiny_base(t5_dir, True)
    data_dir = tmp_path / "summaries"
    prepare(splits, data_dir)
    options.update(
        base_model=str(t5_dir), output=tmp_path / "summarizer", data_dir=data_dir
    )
    summarize(Namespace(**options))
    result = LoraPredictor(options["output"], summarizer=True).generate(
        "def f(x): return x - 1"
    )
    assert result["verified"] is False and result["suggested_fix"] is None
    assert (
        json.loads((options["output"] / "human_review.json").read_text())[0][
            "correctness_0_to_2"
        ]
        is None
    )
