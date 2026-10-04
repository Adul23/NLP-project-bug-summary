"""Score a summarization baseline with corpus BLEU-4 and mean ROUGE-L F1 on one split.

Default split is validation. The test split is for the final evaluation only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.summarization import ast_baseline, bm25_retrieval, tfidf_retrieval
from src.summarization.metrics import corpus_bleu, mean_rouge_l

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data/processed/summarization_dataset.csv"
METHODS = ("ast", "tfidf", "bm25")
_RETRIEVERS = {"tfidf": tfidf_retrieval.retrieve_summaries, "bm25": bm25_retrieval.retrieve_summaries}


def evaluate(dataset: Path, split: str, method: str = "ast") -> dict[str, float | int | str]:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}")

    df = pd.read_csv(dataset, dtype={"source_id": str}, keep_default_na=False)
    subset = df[df["split"] == split]
    if subset.empty:
        raise ValueError(f"No rows for split {split!r}")

    references = subset["summary"].tolist()
    result: dict[str, float | int | str] = {"method": method, "split": split, "examples": len(subset)}
    if method == "ast":
        # Template baseline: no index, no training data.
        hypotheses = [ast_baseline.predict(code) for code in subset["code"]]
    else:
        # Retrieval database is the train split only; the query split is never indexed.
        train = df[df["split"] == "train"]
        if train.empty:
            raise ValueError("No train rows to build the retrieval index")
        hypotheses, no_overlap = _RETRIEVERS[method](
            train["code"].tolist(), train["summary"].tolist(), subset["code"].tolist()
        )
        result["queries_without_token_overlap"] = no_overlap

    result["bleu4"] = round(corpus_bleu(references, hypotheses), 4)
    result["rouge_l_f1"] = round(mean_rouge_l(references, hypotheses), 4)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--method", choices=METHODS, default="ast")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(args.dataset, args.split, args.method)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
