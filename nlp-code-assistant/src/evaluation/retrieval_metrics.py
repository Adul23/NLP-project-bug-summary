"""Ranking metrics for retrieval (error/StackOverflow search, summary retrieval).

Each query has a ranked list of retrieved ids and a set of relevant ids (or graded
relevance for nDCG). All functions are pure Python and deterministic.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be at least 1")


def precision_at_k(ranked: Sequence, relevant: Iterable, k: int) -> float:
    """Share of the top-k results that are relevant (missing positions count as misses)."""
    _check_k(k)
    relevant = set(relevant)
    return sum(1 for item in ranked[:k] if item in relevant) / k


def recall_at_k(ranked: Sequence, relevant: Iterable, k: int) -> float:
    """Share of all relevant items found in the top k (0 when nothing is relevant)."""
    _check_k(k)
    relevant = set(relevant)
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def hit_at_k(ranked: Sequence, relevant: Iterable, k: int) -> float:
    """1.0 if any relevant item is in the top k."""
    _check_k(k)
    relevant = set(relevant)
    return float(any(item in relevant for item in ranked[:k]))


def reciprocal_rank(ranked: Sequence, relevant: Iterable) -> float:
    """1 / rank of the first relevant item, 0 if none is retrieved."""
    relevant = set(relevant)
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def average_precision(ranked: Sequence, relevant: Iterable) -> float:
    """Mean of precision@rank over the ranks of relevant hits, divided by all relevant items."""
    relevant = set(relevant)
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)


def ndcg_at_k(ranked: Sequence, relevance: Mapping | Iterable, k: int) -> float:
    """Normalized discounted cumulative gain with gain (2^rel - 1) and log2(rank + 1) discount.

    relevance is {id: grade} or an iterable of relevant ids (grade 1).
    """
    _check_k(k)
    grades = dict(relevance) if isinstance(relevance, Mapping) else {item: 1 for item in relevance}
    dcg = sum(
        (2 ** grades.get(item, 0) - 1) / math.log2(rank + 1)
        for rank, item in enumerate(ranked[:k], start=1)
    )
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def evaluate_rankings(
    rankings: Sequence[Sequence], relevant: Sequence[Iterable], ks: Sequence[int] = (1, 5, 10)
) -> dict:
    """Mean metrics over queries: P@k, R@k, Hit@k, nDCG@k for each k, plus MRR and MAP."""
    if len(rankings) != len(relevant):
        raise ValueError("rankings and relevant must have one entry per query")
    if not rankings:
        raise ValueError("no queries to evaluate")
    relevant = [set(r) for r in relevant]
    n = len(rankings)
    result = {"queries": n}
    for k in ks:
        result[f"precision@{k}"] = sum(precision_at_k(r, rel, k) for r, rel in zip(rankings, relevant)) / n
        result[f"recall@{k}"] = sum(recall_at_k(r, rel, k) for r, rel in zip(rankings, relevant)) / n
        result[f"hit@{k}"] = sum(hit_at_k(r, rel, k) for r, rel in zip(rankings, relevant)) / n
        result[f"ndcg@{k}"] = sum(ndcg_at_k(r, rel, k) for r, rel in zip(rankings, relevant)) / n
    result["mrr"] = sum(reciprocal_rank(r, rel) for r, rel in zip(rankings, relevant)) / n
    result["map"] = sum(average_precision(r, rel) for r, rel in zip(rankings, relevant)) / n
    return result
