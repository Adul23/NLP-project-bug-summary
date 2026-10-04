"""Deterministic 70/15/15 splitting of whole source-function groups."""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
SPLIT_NAMES = ("train", "validation", "test")


def validate_frame(frame: pd.DataFrame) -> None:
    required = {"source_id", "code", "label", "mutation_type"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(frame.columns))}")
    if frame.empty or not frame["label"].isin([0, 1]).all():
        raise ValueError("Dataset must be nonempty and labels must contain only 0 and 1")
    for column in ("source_id", "code", "mutation_type"):
        if not frame[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"{column} must contain nonempty strings")


def read_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"source_id": str}, keep_default_na=False)
    validate_frame(frame)
    return frame


def split_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ratios apply to unique IDs; row counts vary with mutation availability."""
    validate_frame(frame)
    source_ids = sorted(frame["source_id"].unique())
    if len(source_ids) < 4:
        raise ValueError("At least four source IDs are required for three nonempty splits")
    train_ids, held_out_ids = train_test_split(source_ids, test_size=0.30, random_state=42)
    validation_ids, test_ids = train_test_split(held_out_ids, test_size=0.50, random_state=42)
    splits = tuple(frame.loc[frame["source_id"].isin(ids)].reset_index(drop=True)
                   for ids in (train_ids, validation_ids, test_ids))
    validate_splits(*splits)
    return splits


def validate_splits(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> dict:
    frames = (train, validation, test)
    for frame in frames:
        validate_frame(frame)
        if set(frame["label"]) != {0, 1}:
            raise ValueError("Each split must contain both clean and buggy samples")
    groups = [set(frame["source_id"]) for frame in frames]
    overlap = {"train_validation": len(groups[0] & groups[1]),
               "train_test": len(groups[0] & groups[2]),
               "validation_test": len(groups[1] & groups[2])}
    if any(overlap.values()):
        raise ValueError(f"source_id overlap detected: {overlap}")
    return overlap


def dataset_statistics(frame: pd.DataFrame) -> dict:
    return {"samples": len(frame), "source_ids": int(frame["source_id"].nunique()),
            "clean": int((frame["label"] == 0).sum()), "buggy": int((frame["label"] == 1).sum()),
            "mutation_types": {str(key): int(value) for key, value in
                               frame.loc[frame["label"] == 1, "mutation_type"].value_counts().items()}}


def save_splits(splits: tuple, output_dir: Path) -> None:
    validate_splits(*splits)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in zip(SPLIT_NAMES, splits):
        frame.to_csv(output_dir / f"{name}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/bug_detection_dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    args = parser.parse_args()
    splits = split_dataset(read_dataset(args.input))
    save_splits(splits, args.output_dir)
    print(json.dumps({"splits": {name: dataset_statistics(part) for name, part in zip(SPLIT_NAMES, splits)},
                      "source_id_overlap": validate_splits(*splits)}, indent=2))


if __name__ == "__main__":
    main()
