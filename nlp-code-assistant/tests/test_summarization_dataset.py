"""Tests for src/summarization/build_summary_dataset.py."""

from __future__ import annotations

import argparse
import importlib.util
import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "summarization"
    / "build_summary_dataset.py"
)
_spec = importlib.util.spec_from_file_location("build_summary_dataset", MODULE_PATH)
bsd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bsd)


def summarize(doc):
    return bsd.clean_summary(doc, min_words=3, max_words=30)


# ---------- clean_summary ----------


def test_takes_first_sentence():
    text, reason = summarize("Return the sum of two numbers. Extra detail here.")
    assert (text, reason) == ("Return the sum of two numbers.", "kept")


def test_joins_multiline_first_paragraph_and_drops_rest():
    text, reason = summarize("Return the sum\nof two numbers.\n\nSecond paragraph.")
    assert text == "Return the sum of two numbers."
    assert reason == "kept"


def test_abbreviations_do_not_end_sentence():
    text, _ = summarize("Compute a value, e.g. a sum of items. More text.")
    assert text == "Compute a value, e.g. a sum of items."


def test_rst_roles_and_backticks_removed():
    text, reason = summarize(":class:`Foo` does things with `bar` well.")
    assert text == "Foo does things with bar well."
    assert reason == "kept"


def test_param_and_doctest_docstrings_rejected():
    assert summarize(":param x: some value") == (None, "not_a_description")
    assert summarize(">>> foo(1)") == (None, "not_a_description")
    assert summarize("Args: x is a value") == (None, "not_a_description")


def test_none_and_empty_rejected():
    assert summarize(None) == (None, "no_docstring")
    assert summarize("   ") == (None, "no_docstring")


def test_length_limits():
    assert summarize("Decorator.") == (None, "too_short")
    assert summarize(" ".join(["word"] * 31)) == (None, "too_long")
    assert summarize(" ".join(["word"] * 30))[1] == "kept"


def test_non_english_rejected():
    assert summarize("Возвращает сумму двух чисел")[1] == "non_english"


def test_invalid_unicode_rejected():
    assert summarize("Set the value \ud83d now please")[1] == "invalid_unicode"


# ---------- extract_docstring ----------


def test_extract_docstring_function():
    code = 'def f(x):\n    """Add one to x."""\n    return x + 1\n'
    assert bsd.extract_docstring(code) == "Add one to x."


def test_extract_docstring_async_function():
    code = 'async def f(x):\n    """Fetch x."""\n    return x\n'
    assert bsd.extract_docstring(code) == "Fetch x."


def test_extract_docstring_none_cases():
    assert bsd.extract_docstring("def f(x):\n    return x\n") is None
    assert bsd.extract_docstring("x = 1\n") is None
    assert bsd.extract_docstring("def broken(:\n") is None
    assert bsd.extract_docstring("") is None


def test_invalid_escape_sequences_do_not_raise_or_warn():
    code = 'def f(x):\n    """Match \\w+ tokens in text."""\n    return x\n'
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert bsd.extract_docstring(code) == "Match \\w+ tokens in text."


# ---------- split loading ----------


def write_split(directory: Path, name: str, ids: list[str]) -> None:
    pd.DataFrame({"source_id": ids, "code": "x", "label": 0}).to_csv(
        directory / f"{name}.csv", index=False
    )


def test_load_split_map_keeps_leading_zeros(tmp_path):
    write_split(tmp_path, "train", ["000001"])
    write_split(tmp_path, "validation", ["000002"])
    write_split(tmp_path, "test", ["000003"])
    assert bsd.load_split_map(tmp_path) == {
        "000001": "train",
        "000002": "validation",
        "000003": "test",
    }


def test_load_split_map_detects_overlap(tmp_path):
    write_split(tmp_path, "train", ["000001"])
    write_split(tmp_path, "validation", ["000002"])
    write_split(tmp_path, "test", ["000001"])
    with pytest.raises(ValueError, match="000001"):
        bsd.load_split_map(tmp_path)


def test_load_split_map_missing_file(tmp_path):
    write_split(tmp_path, "train", ["000001"])
    with pytest.raises(FileNotFoundError):
        bsd.load_split_map(tmp_path)


# ---------- end-to-end ----------


def make_args(tmp_path: Path, raw: Path, clean: Path) -> argparse.Namespace:
    return argparse.Namespace(
        raw=raw,
        clean=clean,
        split_dir=tmp_path,
        output=tmp_path / "out" / "summarization_dataset.csv",
        min_words=3,
        max_words=30,
    )


def test_build_end_to_end(tmp_path):
    raw = tmp_path / "raw.jsonl"
    records = [
        {
            "source_id": "000001",
            "code": 'def add(a, b):\n    """Add two numbers together."""\n    return a + b',
        },
        {"source_id": "000002", "code": "def sub(a, b):\n    return a - b"},
        {"source_id": None, "code": "def rejected():\n    pass"},
    ]
    raw.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    clean = tmp_path / "clean.csv"
    pd.DataFrame(
        {
            "source_id": ["000001", "000002"],
            "code": ["def add(a, b):\n    return a + b", "def sub(a, b):\n    return a - b"],
            "label": [0, 0],
            "mutation_type": ["clean", "clean"],
        }
    ).to_csv(clean, index=False)

    write_split(tmp_path, "train", ["000001"])
    write_split(tmp_path, "validation", ["000002"])
    write_split(tmp_path, "test", ["000003"])

    args = make_args(tmp_path, raw, clean)
    bsd.build(args)

    df = pd.read_csv(args.output, dtype={"source_id": str}, keep_default_na=False)
    assert list(df.columns) == ["source_id", "code", "summary", "split"]
    assert len(df) == 1
    row = df.iloc[0]
    assert row["source_id"] == "000001"
    assert row["summary"] == "Add two numbers together."
    assert row["split"] == "train"
    assert '"""' not in row["code"]

    stats = json.loads(args.output.with_suffix(".stats.json").read_text())
    assert stats["accepted_functions"] == 2
    assert stats["kept"] == 1
    assert stats["rejected"] == {"no_docstring": 1}


def test_build_fails_when_source_id_missing_from_split(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        json.dumps(
            {"source_id": "000009", "code": 'def f(x):\n    """Do a thing well."""\n    return x'}
        ),
        encoding="utf-8",
    )
    clean = tmp_path / "clean.csv"
    pd.DataFrame(
        {"source_id": ["000009"], "code": ["def f(x):\n    return x"], "label": [0]}
    ).to_csv(clean, index=False)
    write_split(tmp_path, "train", ["000001"])
    write_split(tmp_path, "validation", ["000002"])
    write_split(tmp_path, "test", ["000003"])

    with pytest.raises(ValueError, match="000009"):
        bsd.build(make_args(tmp_path, raw, clean))
