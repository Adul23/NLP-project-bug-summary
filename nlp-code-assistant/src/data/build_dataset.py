"""Build and validate the clean/synthetic dataset; no modeling or splitting."""

from __future__ import annotations

import argparse
import ast
import json
import random
from dataclasses import asdict
from pathlib import Path

import pandas as pd

if __package__:
    from .mutations import (
        ARITHMETIC,
        BOOLEANS,
        COMPARISONS,
        MUTATION_TYPES,
        MutationStats,
        mutate_code,
    )
    from .preprocess import (
        CLEAN_COLUMNS,
        CLEAN_PATH,
        PROJECT_ROOT,
        SEED,
        is_function,
        parse_python,
        write_csv,
    )
else:
    from mutations import (
        ARITHMETIC,
        BOOLEANS,
        COMPARISONS,
        MUTATION_TYPES,
        MutationStats,
        mutate_code,
    )
    from preprocess import (
        CLEAN_COLUMNS,
        CLEAN_PATH,
        PROJECT_ROOT,
        SEED,
        is_function,
        parse_python,
        write_csv,
    )

DATASET_PATH = PROJECT_ROOT / "data/processed/bug_detection_dataset.csv"
DATASET_COLUMNS = ["sample_id", *CLEAN_COLUMNS]


def _check_single_mutation(original: ast.AST, buggy: ast.AST, category: str) -> None:
    changes = []

    def compare(left, right):
        if type(left) is not type(right):
            changes.append((type(left), type(right)))
        elif isinstance(left, ast.AST):
            for name, value in ast.iter_fields(left):
                compare(value, getattr(right, name))
        elif isinstance(left, list):
            if len(left) != len(right):
                raise ValueError("Mutation changed AST structure")
            for a, b in zip(left, right):
                compare(a, b)
        elif left != right:
            raise ValueError("Mutation changed non-operator AST content")

    compare(original, buggy)
    mapping = {
        "comparison_operator": COMPARISONS,
        "arithmetic_operator": ARITHMETIC,
        "boolean_operator": BOOLEANS,
    }[category]
    if (
        len(changes) != 1
        or changes[0][0] not in mapping
        or mapping[changes[0][0]][0] != changes[0][1]
    ):
        raise ValueError("Expected exactly one supported AST operator change")


def validate_dataset(frame: pd.DataFrame) -> None:
    """Enforce parseability, uniqueness, labels, source linkage and one mutation."""
    if list(frame.columns) != DATASET_COLUMNS or frame.empty:
        raise ValueError(f"Expected nonempty dataset with columns {DATASET_COLUMNS}")
    if frame.isna().any().any():
        raise ValueError("Missing dataset values")
    if not frame["sample_id"].is_unique or frame["sample_id"].tolist() != list(
        range(1, len(frame) + 1)
    ):
        raise ValueError("sample_id must be unique and sequential from 1")
    if (
        not frame["source_id"]
        .map(
            lambda value: isinstance(value, str) and value.isdigit() and len(value) >= 6
        )
        .all()
    ):
        raise ValueError(
            'source_id must be a zero-padded string; use dtype={"source_id": str}'
        )
    if not frame["label"].isin([0, 1]).all():
        raise ValueError("Labels must be 0 or 1")
    if (
        not frame["code"]
        .map(lambda code: isinstance(code, str) and bool(code.strip()))
        .all()
    ):
        raise ValueError("Code must be a nonempty string")
    if frame["code"].duplicated().any():
        raise ValueError("Exact duplicate source code")
    clean = frame[frame["label"] == 0]
    buggy = frame[frame["label"] == 1]
    if (
        clean.empty
        or not clean["source_id"].is_unique
        or not buggy["source_id"].is_unique
    ):
        raise ValueError("Require one clean and at most one buggy row per source_id")
    if not clean["mutation_type"].eq("clean").all():
        raise ValueError("Clean samples must have mutation_type=clean")
    if not buggy["mutation_type"].isin(MUTATION_TYPES).all():
        raise ValueError("Buggy samples need a supported mutation_type")
    if not set(buggy["source_id"]).issubset(set(clean["source_id"])):
        raise ValueError("Buggy sample without matching clean source_id")
    # Cache only clean ASTs; never execute collected code.
    clean_trees = {}
    for row in clean.itertuples(index=False):
        try:
            tree = parse_python(row.code)
        except (SyntaxError, ValueError, RecursionError) as exc:
            raise ValueError(f"Invalid clean source {row.source_id}: {exc}") from exc
        if not is_function(tree):
            raise ValueError(f"Not a standalone function: {row.source_id}")
        clean_trees[row.source_id] = tree
    for row in buggy.itertuples(index=False):
        try:
            tree = parse_python(row.code)
        except (SyntaxError, ValueError, RecursionError) as exc:
            raise ValueError(f"Invalid buggy source {row.source_id}: {exc}") from exc
        _check_single_mutation(clean_trees[row.source_id], tree, row.mutation_type)


def build_dataset(
    clean: pd.DataFrame, seed: int = SEED
) -> tuple[pd.DataFrame, MutationStats]:
    """Keep every clean function and at most one globally unique mutant per source."""
    if list(clean.columns) != CLEAN_COLUMNS or not clean["label"].eq(0).all():
        raise ValueError(f"Expected clean CSV with columns {CLEAN_COLUMNS} and label=0")
    initial = clean.copy()
    initial.insert(0, "sample_id", range(1, len(initial) + 1))
    validate_dataset(initial)
    seen = set(clean["code"])  # Include future clean rows to prevent label collisions.
    rng = random.Random(seed)
    stats = MutationStats()
    rows = []
    for clean_row in clean.to_dict("records"):
        rows.append({"sample_id": len(rows) + 1, **clean_row})
        mutation = mutate_code(clean_row["code"], rng, forbidden=seen, stats=stats)
        if mutation is not None:
            seen.add(mutation.code)
            rows.append(
                {
                    "sample_id": len(rows) + 1,
                    "source_id": clean_row["source_id"],
                    "code": mutation.code,
                    "label": 1,
                    "mutation_type": mutation.mutation_type,
                }
            )
    frame = pd.DataFrame(rows, columns=DATASET_COLUMNS)
    validate_dataset(frame)
    return frame, stats


def format_examples(frame: pd.DataFrame, per_category: int = 5) -> str:
    """Return full original/mutated pairs, preferring short readable functions."""
    clean = frame[frame["label"] == 0].set_index("source_id")["code"].to_dict()
    output = []
    for category in MUTATION_TYPES:
        candidates = frame[frame["mutation_type"] == category]
        candidates = candidates.assign(length=candidates["code"].str.len()).sort_values(
            "length", kind="stable"
        )
        output.append(
            f"\n{category}: showing {min(per_category, len(candidates))} examples"
        )
        for row in candidates.head(per_category).itertuples(index=False):
            output.append(
                "\n"
                + "-" * 32
                + f"\nsource_id: {row.source_id}\nmutation: {category}"
                + f"\n\nORIGINAL:\n{clean[row.source_id]}\n\nBUGGY:\n{row.code}\n"
                + "-" * 32
            )
    return "\n".join(output) + "\n"


def save_dataset(input_path: Path, output_path: Path, seed: int = SEED) -> dict:
    clean = pd.read_csv(input_path, dtype={"source_id": str}, keep_default_na=False)
    frame, stats = build_dataset(clean, seed)
    # build_dataset validates all invariants before any CSV is published.
    write_csv(frame, output_path)
    distribution = {
        kind: int(frame["mutation_type"].eq(kind).sum()) for kind in MUTATION_TYPES
    }
    report = {
        "seed": seed,
        "shape": list(frame.shape),
        "clean_samples": int(frame["label"].eq(0).sum()),
        "buggy_samples": int(frame["label"].eq(1).sum()),
        "mutation_distribution": distribution,
        **asdict(stats),
    }
    output_path.with_suffix(".stats.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    examples = format_examples(frame)
    output_path.with_suffix(".examples.txt").write_text(examples, encoding="utf-8")
    print(
        f"Clean samples: {report['clean_samples']}\nBuggy samples: {report['buggy_samples']}"
    )
    print("\nMutation distribution:")
    for kind, count in distribution.items():
        print(f"{kind}: {count}")
    print(f"\nInvalid mutations skipped: {stats.invalid_mutations_skipped}")
    print(f"Duplicate mutations skipped: {stats.duplicate_mutations_skipped}")
    print(f"Functions without candidates: {stats.no_candidate_functions}")
    print(f"Functions without buggy version: {stats.functions_without_bug}")
    print(f"Total dataset size: {len(frame)}\nSaved: {output_path}")
    print(examples)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=CLEAN_PATH)
    parser.add_argument("--output", type=Path, default=DATASET_PATH)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must differ")
    save_dataset(args.input, args.output, args.seed)


if __name__ == "__main__":
    main()
