"""TF-IDF retrieval baseline: summarize a query function with the summary of its nearest training function.

The index is built only from the training split. Queries are compared by cosine
similarity over TF-IDF vectors of identifier sub-tokens (snake_case and camelCase split).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_CHUNK = 500


def code_tokens(code: str) -> list[str]:
    tokens: list[str] = []
    for identifier in _IDENTIFIER.findall(code):
        for part in identifier.split("_"):
            if part:
                tokens.extend(piece.lower() for piece in _CAMEL.split(part) if piece)
    return tokens


def retrieve_summaries(
    train_codes: Sequence[str],
    train_summaries: Sequence[str],
    query_codes: Sequence[str],
) -> tuple[list[str], int]:
    """Return one retrieved summary per query, and how many queries had no shared tokens."""
    if len(train_codes) != len(train_summaries):
        raise ValueError("train_codes and train_summaries must have the same length")
    if not train_codes:
        raise ValueError("retrieval index is empty")

    vectorizer = TfidfVectorizer(analyzer=code_tokens, sublinear_tf=True)
    index = vectorizer.fit_transform(train_codes)
    queries = vectorizer.transform(query_codes)

    predictions: list[str] = []
    no_overlap = 0
    for start in range(0, queries.shape[0], _CHUNK):
        sims = (queries[start : start + _CHUNK] @ index.T).toarray()
        best = sims.argmax(axis=1)
        no_overlap += int((sims.max(axis=1) == 0).sum())
        predictions.extend(train_summaries[i] for i in best)
    return predictions, no_overlap


def predict_split(train_df, query_df) -> tuple[list[str], int]:
    """Predict summaries for query_df rows using only train_df as the retrieval database."""
    return retrieve_summaries(
        train_df["code"].tolist(),
        train_df["summary"].tolist(),
        query_df["code"].tolist(),
    )
