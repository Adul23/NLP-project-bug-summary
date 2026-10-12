"""One immutable, stratified group split shared by every new experiment."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .split_dataset import (
    ROOT,
    SPLIT_NAMES,
    dataset_statistics,
    read_dataset,
    validate_splits,
)

DEFAULT_SPLITS = ROOT / "data/processed/stratified_v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create_splits(dataset, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Split directory is not empty; use an empty directory to preserve existing experiments"
        )
    frame = read_dataset(dataset)
    if "sample_id" not in frame or not frame.sample_id.is_unique:
        raise ValueError("Require unique sample_id values")
    # Stratify whole groups by their class composition, never split rows of a group.
    # In this dataset these strata are clean-only and clean+buggy pairs.
    groups = frame.groupby("source_id").label.agg(
        lambda x: ",".join(map(str, sorted(x)))
    )
    try:
        train_ids, held_ids = train_test_split(
            groups.index, test_size=0.30, random_state=42, stratify=groups
        )
        val_ids, test_ids = train_test_split(
            held_ids, test_size=0.50, random_state=42, stratify=groups.loc[held_ids]
        )
    except ValueError as exc:
        raise ValueError(
            "Not enough source groups per class-composition stratum for 70/15/15 splitting"
        ) from exc
    parts = [
        frame[frame.source_id.isin(ids)].reset_index(drop=True)
        for ids in (train_ids, val_ids, test_ids)
    ]
    overlaps = validate_splits(*parts)
    output.mkdir(parents=True, exist_ok=True)
    for name, part in zip(SPLIT_NAMES, parts):
        part.to_csv(output / f"{name}.csv", index=False)
    manifest = {
        "seed": 42,
        "method": "source-group class-composition stratification",
        "ratios": [0.70, 0.15, 0.15],
        "dataset_sha256": sha256(dataset),
        "source_id_overlap": overlaps,
        "files": {name: sha256(output / f"{name}.csv") for name in SPLIT_NAMES},
        "statistics": {
            name: dataset_statistics(part) for name, part in zip(SPLIT_NAMES, parts)
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_splits(folder=DEFAULT_SPLITS):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    for name in SPLIT_NAMES:
        if sha256(folder / f"{name}.csv") != manifest["files"][name]:
            raise ValueError(f"{name} split differs from its manifest")
    parts = tuple(read_dataset(folder / f"{name}.csv") for name in SPLIT_NAMES)
    validate_splits(*parts)
    ids = pd.concat(parts).sample_id
    if not ids.is_unique:
        raise ValueError("sample_id values must be globally unique")
    return parts, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "data/processed/bug_detection_dataset.csv"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_SPLITS)
    args = parser.parse_args()
    print(json.dumps(create_splits(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
