"""Pure helpers of scripts/bench_inference.py; no model is loaded."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/bench_inference.py"
spec = importlib.util.spec_from_file_location("bench_inference", SCRIPT)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_median_ms_warms_up_and_syncs():
    calls, syncs = [], []
    ms = bench.median_ms(lambda: calls.append(1), lambda: syncs.append(1),
                         warmup=3, repeats=5)
    assert len(calls) == 8  # warm-up runs are not timed but do run
    assert len(syncs) == 6  # once after warm-up, once per timed run
    assert ms >= 0


def test_agreement():
    result = bench.agreement([0.1, 0.6, 0.9, 0.49], [0.1, 0.4, 0.9, 0.51], threshold=0.5)
    assert result["label_agreement"] == 0.5
    assert result["max_abs_prob_diff"] == pytest.approx(0.2)
    with pytest.raises(ValueError):
        bench.agreement([0.1], [], 0.5)


def test_table_shows_numbers_and_errors():
    rows = [
        {"weights": "adapter", "precision": "fp32", "mode": "eager", "batch1_ms": 8.123,
         "batch16_ms": 40.0, "peak_mem_mb": 512.4, "label_agreement": 1.0,
         "max_abs_prob_diff": 0.0},
        {"weights": "merged", "precision": "fp16", "mode": "compile",
         "error": "RuntimeError: no triton"},
    ]
    text = bench.table(rows).splitlines()
    assert text[0].split()[:3] == ["weights", "prec", "mode"]
    assert "8.12" in text[2] and "512" in text[2]
    assert text[3].startswith("merged") and "error: RuntimeError: no triton" in text[3]
