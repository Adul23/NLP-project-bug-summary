import pandas as pd
import pytest

from src.summarization import review_examples as review


def _frame(rows):
    return pd.DataFrame(rows, columns=["source_id", "reference", "prediction", "rouge_l"])


def test_categorize_groups():
    assert review.categorize("return the user name from the record", "name", 0.1) == "too_short"
    assert review.categorize("return the user name", "return the user name", 1.0) == "good"
    assert review.categorize("return the user name", "delete all cached files now", 0.0) == "bad"
    assert review.categorize("return the user name", "get the current user id", 0.4) == "medium"
    assert review.categorize("return the user name", "get the current user id", 0.2) is None


def test_select_is_deterministic_and_disjoint():
    rows = [
        (f"{i:06d}", "return the user name", "return the user name", 1.0) for i in range(20)
    ] + [(f"{i:06d}", "return the user name", "delete all cached files now", 0.0) for i in range(20, 40)]
    frame = _frame(rows)
    first = review.select_examples(frame, seed=1)
    second = review.select_examples(frame, seed=1)
    assert {k: v["source_id"].tolist() for k, v in first.items()} == {
        k: v["source_id"].tolist() for k, v in second.items()
    }
    assert len(first["good"]) == review.GROUP_SIZES["good"]
    assert len(first["bad"]) == review.GROUP_SIZES["bad"]
    ids = [i for v in first.values() for i in v["source_id"]]
    assert len(ids) == len(set(ids))


def test_small_group_takes_all():
    frame = _frame([("000001", "return the user name", "return the user name", 1.0)])
    assert len(review.select_examples(frame)["good"]) == 1


def test_load_validation_rejects_other_splits(tmp_path):
    dataset = tmp_path / "data.csv"
    dataset.write_text("source_id,code,summary,split\n000001,def f(): pass,x y z,validation\n000002,def g(): pass,x y z,test\n")
    predictions = tmp_path / "pred.csv"
    predictions.write_text("source_id,reference,prediction,rouge_l\n000002,a,b,0.0\n")
    with pytest.raises(ValueError):
        review.load_validation(predictions, dataset)
    predictions.write_text("source_id,reference,prediction,rouge_l\n000001,a,b,0.0\n")
    assert review.load_validation(predictions, dataset)["code"].tolist() == ["def f(): pass"]
