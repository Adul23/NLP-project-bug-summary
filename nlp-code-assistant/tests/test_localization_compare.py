import pandas as pd
import pytest
from src.evaluation.localization_compare import compare


def rows(ids, top1):
    return pd.DataFrame(
        {"sample_id": ids, "top1": top1, "top3": [True] * len(ids), "line_top1": top1}
    )


def test_counts_discordant_pairs():
    a = rows([1, 2, 3, 4, 5], [True, True, True, False, True])
    b = rows([1, 2, 3, 4, 5], [True, False, False, True, True])
    result = compare(a, b)
    assert result["n"] == 5
    top1 = result["top1"]
    assert (top1["b"], top1["c"]) == (2, 1)  # only A right twice, only B right once
    assert top1["codebert"] == pytest.approx(0.8)
    assert top1["tfidf"] == pytest.approx(0.6)
    assert 0 < top1["exact_p"] <= 1
    assert result["top3"]["exact_p"] == 1.0  # identical columns


def test_rejects_different_mutants():
    with pytest.raises(ValueError, match="different"):
        compare(rows([1, 2], [True, True]), rows([2, 1], [True, True]))
