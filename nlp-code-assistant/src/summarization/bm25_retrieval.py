"""BM25 retrieval baseline: summarize a query function with the summary of its best-scoring training function.

Same index and tokenizer as the TF-IDF baseline (identifier sub-tokens); only the scoring differs.
Okapi BM25 with k1=1.5 and b=0.75, IDF with the non-negative variant log(1 + (N - df + 0.5) / (df + 0.5)).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from src.summarization.tfidf_retrieval import code_tokens

K1 = 1.5
B = 0.75
_CHUNK = 500


def retrieve_summaries(
    train_codes: Sequence[str],
    train_summaries: Sequence[str],
    query_codes: Sequence[str],
) -> tuple[list[str], int]:
    """Return one retrieved summary per query, and how many queries matched no indexed token."""
    if len(train_codes) != len(train_summaries):
        raise ValueError("train_codes and train_summaries must have the same length")
    if not train_codes:
        raise ValueError("retrieval index is empty")

    vectorizer = CountVectorizer(analyzer=code_tokens)
    tf = vectorizer.fit_transform(train_codes).tocsr().astype(np.float64)
    n_docs = tf.shape[0]

    doc_len = np.asarray(tf.sum(axis=1)).ravel()
    avg_len = doc_len.mean() if doc_len.size else 0.0
    if avg_len == 0:
        raise ValueError("training codes contain no tokens")

    df = np.bincount(tf.indices, minlength=tf.shape[1]).astype(np.float64)
    idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    # BM25 weight per (doc, term): idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avg_len))
    tf = tf.tocoo()
    norm = K1 * (1.0 - B + B * doc_len[tf.row] / avg_len)
    weights = idf[tf.col] * tf.data * (K1 + 1.0) / (tf.data + norm)
    index = sparse.csr_matrix((weights, (tf.row, tf.col)), shape=tf.shape)

    queries = vectorizer.transform(query_codes).tocsr()
    queries.data[:] = 1.0  # each distinct query term counts once

    predictions: list[str] = []
    no_overlap = 0
    for start in range(0, queries.shape[0], _CHUNK):
        scores = (queries[start : start + _CHUNK] @ index.T).toarray()
        best = scores.argmax(axis=1)
        no_overlap += int((scores.max(axis=1) == 0).sum())
        predictions.extend(train_summaries[i] for i in best)
    return predictions, no_overlap
