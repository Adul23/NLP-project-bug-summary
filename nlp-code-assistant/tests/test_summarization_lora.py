"""Fast tests for the LoRA path: no model download, no training run."""

import json
import sys

import pytest

from src.summarization import evaluate_baseline, evaluate_model, final_test_guard
from src.summarization import summarize as summ
from src.summarization import train_codet5, train_codet5_lora
from src.summarization.model_loading import artifact_size_mb, is_full_checkpoint, is_lora_checkpoint

peft = pytest.importorskip("peft")


def tiny_t5():
    from transformers import T5Config, T5ForConditionalGeneration

    return T5ForConditionalGeneration(
        T5Config(vocab_size=64, d_model=16, d_kv=8, d_ff=32, num_layers=1,
                 num_decoder_layers=1, num_heads=2)
    )


# ---- LoRA configuration
def test_default_lora_config():
    cfg = train_codet5_lora.build_lora_config()
    assert cfg.r == 8 and cfg.lora_alpha == 16 and cfg.lora_dropout == 0.1
    assert set(cfg.target_modules) == {"q", "v"}
    assert cfg.task_type == peft.TaskType.SEQ_2_SEQ_LM


@pytest.mark.parametrize(
    "targets,expected",
    [("qv", {"q", "v"}), ("qkvo", {"q", "k", "v", "o"}), ("all-linear", "all-linear")],
)
def test_lora_targets(targets, expected):
    cfg = train_codet5_lora.build_lora_config(r=4, alpha=8, targets=targets)
    assert cfg.r == 4 and cfg.lora_alpha == 8
    got = cfg.target_modules
    assert (set(got) if isinstance(got, (set, list)) else got) == expected


def test_invalid_lora_options():
    with pytest.raises(ValueError):
        train_codet5_lora.build_lora_config(targets="everything")
    with pytest.raises(ValueError):
        train_codet5_lora.build_lora_config(r=0)


def test_lora_freezes_base_and_trains_few_parameters():
    model = peft.get_peft_model(tiny_t5(), train_codet5_lora.build_lora_config())
    trainable, total = train_codet5_lora.count_parameters(model)
    assert 0 < trainable < total * 0.2
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n for n in names)


def test_cli_defaults(monkeypatch):
    seen = {}
    monkeypatch.setattr(train_codet5, "train", lambda args, **kw: seen.update(args=args, **kw))
    monkeypatch.setattr(sys, "argv", ["x"])
    train_codet5_lora.main()
    args = seen["args"]
    assert (args.lora_r, args.lora_alpha, args.lora_targets) == (8, 16, "qv")
    assert args.lr == 2e-4 and args.batch_size == 8 and args.max_length == 384
    assert args.max_target == 48 and args.patience == 2 and args.epochs == 8 and args.seed == 42
    assert seen["default_output"].name == "codet5_summarizer_lora"
    assert seen["extra_settings"]["method"] == "lora"


# ---- output folder
def test_non_empty_output_folder_is_refused(tmp_path):
    (tmp_path / "old.txt").write_text("x")
    args = train_codet5.build_parser("d", 2e-4, "h").parse_args(["--output", str(tmp_path)])
    with pytest.raises(ValueError, match="not empty"):
        train_codet5.train(args, load_model=lambda device: pytest.fail("model must not load"))


def test_smoke_runs_use_separate_folder():
    args = train_codet5.build_parser("d", 2e-4, "h").parse_args(["--limit", "8"])
    out = train_codet5.resolve_output(args, train_codet5_lora.DEFAULT_OUTPUT, "codet5_lora_smoke")
    assert out.name.startswith("codet5_lora_smoke_")
    args = train_codet5.build_parser("d", 2e-4, "h").parse_args([])
    assert train_codet5.resolve_output(args, train_codet5_lora.DEFAULT_OUTPUT) == (
        train_codet5_lora.DEFAULT_OUTPUT
    )


# ---- test-split guard
def test_guard_function():
    final_test_guard.require_test_confirmation("validation", False)
    final_test_guard.require_test_confirmation("test", True)
    with pytest.raises(ValueError, match="--confirm-final-test"):
        final_test_guard.require_test_confirmation("test", False)


def test_evaluate_model_refuses_test_without_flag(capsys):
    with pytest.raises(SystemExit):
        evaluate_model.parse_args(["--split", "test"])
    assert "--confirm-final-test" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        evaluate_model.parse_args(["--split", "test", "--compare"])
    args = evaluate_model.parse_args(["--split", "test", "--confirm-final-test"])
    assert args.split == "test"
    assert "WARNING" in capsys.readouterr().out
    assert evaluate_model.parse_args([]).split == "validation"


def test_evaluate_baseline_refuses_test_without_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--split", "test"])
    with pytest.raises(SystemExit):
        evaluate_baseline.parse_args()


# ---- backend selection in summarize()
def make_full(path):
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")


def make_lora(path):
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "b"}))
    (path / "adapter_model.safetensors").write_bytes(b"x" * 2_000_000)
    (path / "vocab.json").write_bytes(b"y" * 5_000_000)


@pytest.fixture
def folders(tmp_path, monkeypatch):
    monkeypatch.delenv(summ.CHECKPOINT_ENV, raising=False)
    full, lora = tmp_path / "full", tmp_path / "lora"
    monkeypatch.setattr(summ, "DEFAULT_CHECKPOINT", full)
    monkeypatch.setattr(summ, "DEFAULT_LORA_CHECKPOINT", lora)
    monkeypatch.setattr(summ, "DEFAULT_LORA_R16_CHECKPOINT", tmp_path / "lora_r16")
    return full, lora


def test_auto_prefers_existing_checkpoint(folders):
    full, lora = folders
    assert summ.resolve_checkpoint() == full  # nothing exists: loader explains
    make_lora(lora)
    assert summ.resolve_checkpoint() == lora
    make_full(full)
    assert summ.resolve_checkpoint() == full


def test_explicit_backend_and_checkpoint(folders, monkeypatch):
    full, lora = folders
    assert summ.resolve_checkpoint(backend="lora") == lora
    assert summ.resolve_checkpoint(backend="full") == full
    assert summ.resolve_checkpoint("/x", backend="lora") == summ.Path("/x")
    monkeypatch.setenv(summ.CHECKPOINT_ENV, "/env")
    assert summ.resolve_checkpoint(backend="lora") == summ.Path("/env")
    with pytest.raises(ValueError):
        summ.resolve_checkpoint(backend="gpt")


def test_lora_backend_prefers_r16_then_r8(folders, tmp_path):
    full, lora = folders
    r16 = tmp_path / "lora_r16"
    assert summ.resolve_checkpoint(backend="lora") == lora  # no r16: r8 folder
    make_lora(lora)
    assert summ.resolve_checkpoint(backend="lora") == lora
    make_lora(r16)
    assert summ.resolve_checkpoint(backend="lora") == r16
    assert summ.resolve_checkpoint() == r16  # auto without full: the preferred LoRA
    make_full(full)
    assert summ.resolve_checkpoint() == full  # auto keeps preferring full
    assert summ.resolve_checkpoint(backend="full") == full


def test_missing_checkpoints_error_is_clear(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_codet5_lora"):
        summ._HfBackend(tmp_path, None)


def test_summarizer_passes_backend_to_resolution(folders, monkeypatch):
    _, lora = folders
    seen = []

    class Fake:
        def __init__(self, checkpoint, device):
            seen.append(checkpoint)

        def generate(self, code, **kw):
            return "ok"

    monkeypatch.setattr(summ, "_backend_factory", Fake)
    summ._SUMMARIZERS.clear()
    assert summ.summarize("def f():\n    pass", backend="lora") == "ok"
    assert seen == [lora]


# ---- checkpoint kinds and sizes
def test_checkpoint_kinds_and_artifact_size(tmp_path):
    make_full(tmp_path / "f")
    make_lora(tmp_path / "l")
    assert is_full_checkpoint(tmp_path / "f") and not is_lora_checkpoint(tmp_path / "f")
    assert is_lora_checkpoint(tmp_path / "l") and not is_full_checkpoint(tmp_path / "l")
    # LoRA size counts adapter files only (not the tokenizer vocab)
    assert artifact_size_mb(tmp_path / "l") == pytest.approx(2.0, abs=0.01)


# ---- comparison table
def test_comparison_table_marks_missing_values():
    rows = [
        {"label": "AST template", "bleu4": 0.0091, "rouge_l_f1": 0.2252,
         "trainable_parameters": None},
        {"label": "CodeT5-small LoRA", "bleu4": "TBD", "rouge_l_f1": "TBD",
         "latency_ms_median": "TBD", "latency_ms_mean": "TBD",
         "artifact_size_mb": "TBD", "trainable_parameters": "TBD"},
        {"label": "CodeT5-small full FT", "bleu4": 0.139, "rouge_l_f1": 0.3796,
         "latency_ms_median": 120.0, "latency_ms_mean": 130.5,
         "artifact_size_mb": 242.1, "trainable_parameters": 60517632},
    ]
    text = evaluate_model.render_comparison(rows, "validation", 1384)
    assert "| AST template | 0.0091 | 0.2252 | — | — | — | — |" in text
    assert "| CodeT5-small LoRA | TBD | TBD | TBD | TBD | TBD | TBD |" in text
    assert "60,517,632" in text and "1384 examples" in text
