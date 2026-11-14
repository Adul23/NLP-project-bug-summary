from pathlib import Path

import pytest

from src.summarization import summarize as mod


class FakeBackend:
    created = 0

    def __init__(self, checkpoint, device):
        type(self).created += 1
        self.checkpoint = checkpoint
        self.calls = []

    def generate(self, code, **kwargs):
        self.calls.append((code, kwargs))
        return "  Add two\n numbers.  "


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    FakeBackend.created = 0
    mod._SUMMARIZERS.clear()
    monkeypatch.setattr(mod, "_backend_factory", FakeBackend)
    monkeypatch.delenv(mod.CHECKPOINT_ENV, raising=False)


def test_lazy_loading_and_single_load():
    s = mod.CodeSummarizer("some/folder")
    assert not s.loaded and FakeBackend.created == 0
    assert s.summarize("def add(a, b):\n    return a + b") == "Add two numbers."
    s.summarize("def f():\n    pass")
    assert s.loaded and FakeBackend.created == 1


def test_generation_settings_are_passed():
    s = mod.CodeSummarizer("x")
    s.summarize("def f():\n    pass")
    code, kwargs = s._backend.calls[0]
    assert kwargs == {
        "num_beams": 4, "length_penalty": 2.0, "max_length": 384, "max_new_tokens": 48,
    }


def test_checkpoint_resolution(monkeypatch):
    assert mod.resolve_checkpoint() == mod.DEFAULT_CHECKPOINT
    monkeypatch.setenv(mod.CHECKPOINT_ENV, "/from/env")
    assert mod.resolve_checkpoint() == Path("/from/env")
    assert mod.resolve_checkpoint("/explicit") == Path("/explicit")


@pytest.mark.parametrize("bad", ["", "   \n", None, 5, "x" * (mod.MAX_CODE_CHARS + 1)])
def test_invalid_code_rejected_before_loading(bad):
    s = mod.CodeSummarizer("x")
    with pytest.raises(ValueError):
        s.summarize(bad)
    assert FakeBackend.created == 0


def test_module_function_caches_per_checkpoint():
    mod.summarize("def f():\n    pass", "folder_a")
    mod.summarize("def g():\n    pass", "folder_a")
    mod.summarize("def h():\n    pass", "folder_b")
    assert FakeBackend.created == 2


def test_missing_checkpoint_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="CODET5_SUMMARIZER_CHECKPOINT"):
        mod._HfBackend(tmp_path, None)
