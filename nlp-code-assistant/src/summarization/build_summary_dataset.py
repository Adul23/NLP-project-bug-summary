"""Build the code-summarization dataset.

For every accepted function the target summary is the cleaned first sentence
of the original docstring (taken from the raw CodeSearchNet subset). The model
input is the clean function code WITHOUT the docstring (from
clean_functions.csv). Train/validation/test assignment is copied from the
bug-detection split (grouped by source_id), so both parts of the project share
the same partition and never leak across splits.

Output: data/processed/summarization_dataset.csv
        columns: source_id, code, summary, split
Dataset code is parsed but never executed.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = PROJECT_ROOT / "data/raw/codesearchnet/python_functions.jsonl"
DEFAULT_CLEAN = PROJECT_ROOT / "data/processed/clean_functions.csv"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data/processed"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/summarization_dataset.csv"

SPLIT_NAMES = ("train", "validation", "test")

_ABBREVIATIONS = {
    "e.g.": "e<dot>g<dot>",
    "i.e.": "i<dot>e<dot>",
    "etc.": "etc<dot>",
    "vs.": "vs<dot>",
    "cf.": "cf<dot>",
}
_ROLE = re.compile(r":[a-zA-Z]+:`~?([^`]+)`")
_URL = re.compile(r"https?://\S+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_NOT_A_DESCRIPTION = re.compile(
    r"^(:param|:return|:rtype|@|>>>|\.\.\s|args?:|returns?:|todo|fixme)",
    re.IGNORECASE,
)


def extract_docstring(code: str) -> str | None:
    """Return the cleaned docstring of the single top-level function."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError):
            return None
    if not tree.body:
        return None
    node = tree.body[0]
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    return ast.get_docstring(node, clean=True)


def clean_summary(
    doc: str | None, min_words: int, max_words: int
) -> tuple[str | None, str]:
    """Return (summary, reason). Summary is None when the example is rejected."""
    if not doc or not doc.strip():
        return None, "no_docstring"

    paragraph = doc.strip().split("\n\n")[0]
    text = " ".join(paragraph.split())
    if _NOT_A_DESCRIPTION.match(text):
        return None, "not_a_description"

    for abbreviation, placeholder in _ABBREVIATIONS.items():
        text = text.replace(abbreviation, placeholder)
    text = _SENTENCE_END.split(text, maxsplit=1)[0]
    text = text.replace("<dot>", ".")
    text = _ROLE.sub(r"\1", text)
    text = _URL.sub("", text).replace("`", "")
    text = " ".join(text.split()).strip()

    if not text or not text[0].isalpha():
        return None, "not_a_description"
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return None, "invalid_unicode"
    if sum(ch.isascii() for ch in text) / len(text) < 0.9:
        return None, "non_english"

    n_words = len(text.split())
    if n_words < min_words:
        return None, "too_short"
    if n_words > max_words:
        return None, "too_long"
    return text, "kept"


def load_split_map(split_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name in SPLIT_NAMES:
        path = split_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run: python src/bug_detection/split_dataset.py"
            )
        ids = pd.read_csv(
            path,
            dtype={"source_id": str},
            usecols=["source_id"],
            keep_default_na=False,
        )["source_id"]
        for source_id in set(ids):
            if source_id in mapping:
                raise ValueError(
                    f"source_id {source_id} appears in both "
                    f"{mapping[source_id]} and {name}"
                )
            mapping[source_id] = name
    return mapping


def load_clean_code(path: Path) -> dict[str, str]:
    df = pd.read_csv(path, dtype={"source_id": str}, keep_default_na=False)
    if df["source_id"].duplicated().any():
        raise ValueError("clean_functions.csv has duplicate source_id values")
    return dict(zip(df["source_id"], df["code"]))


def build(args: argparse.Namespace) -> None:
    split_map = load_split_map(args.split_dir)
    clean_code = load_clean_code(args.clean)

    rows = []
    reasons: Counter[str] = Counter()
    total = 0

    with args.raw.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            source_id = record.get("source_id")
            if source_id is None:
                continue
            total += 1

            if source_id not in clean_code:
                raise ValueError(f"source_id {source_id} missing in clean CSV")
            if source_id not in split_map:
                raise ValueError(f"source_id {source_id} missing in split files")

            doc = extract_docstring(record["code"])
            summary, reason = clean_summary(doc, args.min_words, args.max_words)
            reasons[reason] += 1
            if summary is None:
                continue

            code = clean_code[source_id]
            if not code.strip():
                raise ValueError(f"empty code for source_id {source_id}")
            rows.append(
                {
                    "source_id": source_id,
                    "code": code,
                    "summary": summary,
                    "split": split_map[source_id],
                }
            )

    if not rows:
        raise RuntimeError("No summarization examples were produced")

    df = pd.DataFrame(rows).sort_values("source_id").reset_index(drop=True)
    if df["source_id"].duplicated().any():
        raise RuntimeError("Duplicate source_id in output")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    words = df["summary"].str.split().str.len()
    stats = {
        "accepted_functions": total,
        "kept": len(df),
        "rejected": {k: v for k, v in reasons.items() if k != "kept"},
        "split_counts": df["split"].value_counts().to_dict(),
        "summary_words": {
            "min": int(words.min()),
            "median": float(words.median()),
            "mean": round(float(words.mean()), 2),
            "max": int(words.max()),
        },
        "duplicate_summaries": int(df["summary"].duplicated().sum()),
        "min_words": args.min_words,
        "max_words": args.max_words,
    }
    stats_path = args.output.with_suffix(".stats.json")
    stats_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Accepted functions: {total}")
    print(f"Kept: {len(df)}")
    for reason, count in reasons.items():
        if reason != "kept":
            print(f"Rejected ({reason}): {count}")
    print("Split counts:", stats["split_counts"])
    print("Summary words:", stats["summary_words"])
    print(f"Duplicate summaries: {stats['duplicate_summaries']}")
    print(f"Saved: {args.output}")
    print(f"Stats: {stats_path}")

    print("\nExamples:")
    for _, row in df.sample(n=min(5, len(df)), random_state=42).iterrows():
        print("--------------------------------")
        print(f"source_id: {row['source_id']} | split: {row['split']}")
        print(f"SUMMARY: {row['summary']}")
        print("CODE:")
        print(row["code"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--clean", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-words", type=int, default=3)
    parser.add_argument("--max-words", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())