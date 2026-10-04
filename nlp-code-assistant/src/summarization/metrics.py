"""Dependency-free BLEU-4 and ROUGE-L for summary evaluation.

Tokenization is lowercase word/number tokens, so scores do not depend on
punctuation or capitalization.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(references: Sequence[str], hypotheses: Sequence[str], max_n: int = 4) -> float:
    """Corpus-level BLEU (Papineni et al., 2002), uniform weights, brevity penalty.

    Returns a value in [0, 1]. Returns 0.0 if any n-gram order has no matches.
    """
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")

    matches = [0] * max_n
    totals = [0] * max_n
    ref_len = 0
    hyp_len = 0
    for ref_text, hyp_text in zip(references, hypotheses):
        ref = tokenize(ref_text)
        hyp = tokenize(hyp_text)
        ref_len += len(ref)
        hyp_len += len(hyp)
        for n in range(1, max_n + 1):
            hyp_counts = _ngrams(hyp, n)
            ref_counts = _ngrams(ref, n)
            matches[n - 1] += sum(min(c, ref_counts[g]) for g, c in hyp_counts.items())
            totals[n - 1] += max(len(hyp) - n + 1, 0)

    if hyp_len == 0 or min(matches) == 0 or min(totals) == 0:
        return 0.0

    log_precision = sum(math.log(m / t) for m, t in zip(matches, totals)) / max_n
    brevity = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return brevity * math.exp(log_precision)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0]
        for j, y in enumerate(b, start=1):
            current.append(previous[j - 1] + 1 if x == y else max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]


def rouge_l_f1(reference: str, hypothesis: str) -> float:
    """Sentence-level ROUGE-L F1 based on the longest common subsequence."""
    ref = tokenize(reference)
    hyp = tokenize(hypothesis)
    if not ref or not hyp:
        return 0.0
    lcs = _lcs_length(ref, hyp)
    if lcs == 0:
        return 0.0
    precision = lcs / len(hyp)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def mean_rouge_l(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    if not references:
        raise ValueError("no examples to score")
    return sum(rouge_l_f1(r, h) for r, h in zip(references, hypotheses)) / len(references)
