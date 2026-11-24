"""Pick validation predictions for manual error analysis and write a readable report.

Reads predictions_validation.csv (made by evaluate_model.py) and attaches the code
of those validation rows from summarization_dataset.csv. The test split is never read:
the script refuses predictions whose source_id is not in the validation split.

Groups: good, medium, bad, too_short (a short prediction goes only to too_short).
Sampling is random but fixed by --seed. Nothing is generated; no model is loaded.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import pandas as pd

from src.summarization.metrics import tokenize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data/processed/summarization_dataset.csv"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "checkpoints/codet5_summarizer/predictions_validation.csv"

GOOD_MIN = 0.6
MEDIUM_MIN = 0.3
MEDIUM_MAX = 0.6
BAD_MAX = 0.15
SHORT_WORDS = 3
GROUP_SIZES = {"good": 6, "medium": 6, "bad": 8, "too_short": 6}
GROUP_ORDER = ("good", "medium", "bad", "too_short")


def categorize(reference: str, prediction: str, rouge_l: float) -> str | None:
    """Return the review group of one example, or None if it fits no group."""
    ref_words = len(tokenize(reference))
    pred_words = len(tokenize(prediction))
    if pred_words <= SHORT_WORDS or pred_words * 2 <= ref_words:
        return "too_short"
    if rouge_l >= GOOD_MIN:
        return "good"
    if rouge_l < BAD_MAX:
        return "bad"
    if MEDIUM_MIN <= rouge_l < MEDIUM_MAX:
        return "medium"
    return None


def select_examples(frame: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Group the rows and sample each group; every row appears in at most one group."""
    groups = {name: [] for name in GROUP_ORDER}
    for row in frame.sort_values("source_id").itertuples(index=False):
        name = categorize(row.reference, row.prediction, row.rouge_l)
        if name:
            groups[name].append(row.source_id)
    rng = random.Random(seed)
    selected = {}
    for name in GROUP_ORDER:
        ids = groups[name]
        take = min(GROUP_SIZES[name], len(ids))
        chosen = sorted(rng.sample(ids, take))
        selected[name] = frame[frame["source_id"].isin(chosen)].sort_values("source_id")
    return selected


def group_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = Counter(
        categorize(r.reference, r.prediction, r.rouge_l) or "other"
        for r in frame.itertuples(index=False)
    )
    return {name: counts.get(name, 0) for name in (*GROUP_ORDER, "other")}


def load_validation(predictions_path: Path, dataset_path: Path) -> pd.DataFrame:
    """Predictions joined with their code; fails if any id is outside validation."""
    preds = pd.read_csv(predictions_path, dtype={"source_id": str}, keep_default_na=False)
    data = pd.read_csv(
        dataset_path, usecols=["source_id", "code", "split"],
        dtype={"source_id": str}, keep_default_na=False,
    )
    validation = data[data["split"] == "validation"][["source_id", "code"]]
    unknown = set(preds["source_id"]) - set(validation["source_id"])
    if unknown:
        raise ValueError(
            f"{len(unknown)} predictions are not in the validation split, e.g. {sorted(unknown)[:3]}"
        )
    return preds.merge(validation, on="source_id", how="left")


def statistics_text(frame: pd.DataFrame) -> str:
    scores = frame["rouge_l"]
    lines = [
        f"examples: {len(frame)}",
        f"mean ROUGE-L: {scores.mean():.4f} | median: {scores.median():.4f}",
        "ROUGE-L buckets: "
        + ", ".join(
            f"{label} {int(mask.sum())}"
            for label, mask in (
                ("=0", scores == 0),
                ("0-0.3", (scores > 0) & (scores < 0.3)),
                ("0.3-0.6", (scores >= 0.3) & (scores < 0.6)),
                (">=0.6", scores >= 0.6),
            )
        ),
        "groups (all validation): " + ", ".join(f"{k} {v}" for k, v in group_counts(frame).items()),
    ]
    first_words = Counter(
        (tokenize(p) or ["<empty>"])[0] for p in frame["prediction"]
    ).most_common(8)
    lines.append("most common first words of predictions: " + ", ".join(f"{w} ({n})" for w, n in first_words))
    repeated = Counter(frame["prediction"].str.lower()).most_common(5)
    lines.append("most repeated predictions: " + "; ".join(f"'{t}' x{n}" for t, n in repeated))
    return "\n".join(f"- {line}" for line in lines)


def render(selected: dict[str, pd.DataFrame], frame: pd.DataFrame, max_code_lines: int) -> str:
    parts = ["# Manual review of validation predictions\n", "## Statistics\n", statistics_text(frame), ""]
    for name in GROUP_ORDER:
        rows = selected[name]
        parts.append(f"\n## {name} ({len(rows)} examples)\n")
        for row in rows.itertuples(index=False):
            code_lines = row.code.splitlines()
            shown = "\n".join(code_lines[:max_code_lines])
            if len(code_lines) > max_code_lines:
                shown += f"\n# ... {len(code_lines) - max_code_lines} more lines"
            parts.append(
                f"### {name} | source_id {row.source_id} | ROUGE-L {row.rouge_l:.2f}\n\n"
                f"- reference: {row.reference}\n- prediction: {row.prediction}\n\n"
                f"~~~~python\n{shown}\n~~~~\n"
            )
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=None,
                        help="default: review_validation.md next to the predictions file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-code-lines", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.predictions.parent / "review_validation.md"
    if output.exists():
        raise ValueError(f"{output} already exists; choose another --output")
    frame = load_validation(args.predictions, args.dataset)
    selected = select_examples(frame, args.seed)
    output.write_text(render(selected, frame, args.max_code_lines))
    print(f"saved: {output}")
    print(statistics_text(frame))


if __name__ == "__main__":
    main()
