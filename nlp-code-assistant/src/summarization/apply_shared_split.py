"""Assign the summarization dataset to the team's shared split (stratified_v1).

Only the `split` column changes; code and summaries stay as they are.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NAMES = ("train", "validation", "test")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_split_ids(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    ids = {}
    for name in NAMES:
        if sha256(folder / f"{name}.csv") != manifest["files"][name]:
            raise ValueError(f"{name}.csv differs from its manifest")
        frame = pd.read_csv(
            folder / f"{name}.csv", dtype={"source_id": str}, keep_default_na=False
        )
        ids[name] = set(frame["source_id"])
    for i, a in enumerate(NAMES):
        for b in NAMES[i + 1 :]:
            if ids[a] & ids[b]:
                raise ValueError(f"source_id overlap between {a} and {b}")
    return ids


def apply_split(dataset, ids):
    lookup = {sid: name for name, group in ids.items() for sid in group}
    result = dataset.copy()
    result["split"] = result["source_id"].map(lookup)
    if result["split"].isna().any():
        missing = result.loc[result["split"].isna(), "source_id"].tolist()[:5]
        raise ValueError(f"source_id not found in shared split, e.g. {missing}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "data/processed/summarization_dataset.csv"
    )
    parser.add_argument(
        "--split-dir", type=Path, default=ROOT / "data/processed/stratified_v1"
    )
    args = parser.parse_args()

    dataset = pd.read_csv(args.dataset, dtype={"source_id": str}, keep_default_na=False)
    updated = apply_split(dataset, load_split_ids(args.split_dir))
    updated = updated[["source_id", "code", "summary", "split"]]
    updated.to_csv(args.dataset, index=False)
    print("rows:", len(updated))
    print(updated["split"].value_counts().to_dict())


if __name__ == "__main__":
    main()