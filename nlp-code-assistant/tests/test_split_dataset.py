"""Source grouping is the leakage boundary, even for mutated functions."""
import pandas as pd
import pytest


def paired_frame():
    return pd.DataFrame([
        {"source_id": f"{i:06d}", "code": f"def f{i}(x): return x {'+' if label == 0 else '-'} 1",
         "label": label, "mutation_type": "clean" if label == 0 else "arithmetic_operator"}
        for i in range(100) for label in (0, 1)
    ])


def test_source_groups_are_disjoint_complete_and_repeatable():
    from src.bug_detection.split_dataset import split_dataset, validate_splits
    frame = paired_frame()
    splits = split_dataset(frame)
    validate_splits(*splits)
    assert [len(part) for part in splits] == [140, 30, 30]
    assert [part.source_id.nunique() for part in splits] == [70, 15, 15]
    groups = [set(part.source_id) for part in splits]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert set.union(*groups) == set(frame.source_id)
    for part, repeated in zip(splits, split_dataset(frame)):
        assert set(part.label) == {0, 1}
        pd.testing.assert_frame_equal(part, repeated)


@pytest.mark.parametrize("column,value", [("label", 2), ("label", None),
                                          ("code", ""), ("code", None),
                                          ("source_id", ""), ("source_id", None),
                                          ("mutation_type", None)])
def test_invalid_data_is_rejected(column, value):
    from src.bug_detection.split_dataset import split_dataset
    frame = paired_frame()
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        split_dataset(frame)


def test_leaking_splits_are_rejected():
    from src.bug_detection.split_dataset import split_dataset, validate_splits
    train, validation, test = split_dataset(paired_frame())
    with pytest.raises(ValueError, match="overlap"):
        validate_splits(train, pd.concat([validation, train.iloc[:1]]), test)


def test_csv_preserves_leading_zero_source_ids(tmp_path):
    from src.bug_detection.split_dataset import read_dataset, save_splits, split_dataset
    splits = split_dataset(paired_frame())
    save_splits(splits, tmp_path)
    for name, part in zip(("train", "validation", "test"), splits):
        loaded = read_dataset(tmp_path / f"{name}.csv")
        pd.testing.assert_frame_equal(loaded, part)
        assert loaded.source_id.str.len().eq(6).all()
