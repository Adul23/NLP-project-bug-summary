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


# --- extended metrics (no network access; METEOR only with locally installed WordNet) ---


def _bleu_stats(reference: str, hypothesis: str, max_n: int = 4):
    ref, hyp = tokenize(reference), tokenize(hypothesis)
    matches, totals = [], []
    for n in range(1, max_n + 1):
        hyp_counts, ref_counts = _ngrams(hyp, n), _ngrams(ref, n)
        matches.append(sum(min(c, ref_counts[g]) for g, c in hyp_counts.items()))
        totals.append(max(len(hyp) - n + 1, 0))
    return matches, totals, len(ref), len(hyp)


def _bleu_from_stats(matches, totals, ref_len, hyp_len) -> float:
    if hyp_len == 0 or min(matches) == 0 or min(totals) == 0:
        return 0.0
    log_precision = sum(math.log(m / t) for m, t in zip(matches, totals)) / len(matches)
    brevity = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return brevity * math.exp(log_precision)


def sentence_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """Smoothed sentence BLEU-4 (Lin & Och 2004: add one to n-gram counts for n >= 2).

    This is the "smoothed BLEU" used in CodeSearchNet/CodeXGLUE code summarization.
    """
    matches, totals, ref_len, hyp_len = _bleu_stats(reference, hypothesis, max_n)
    if hyp_len == 0:
        return 0.0
    log_precision = 0.0
    for n, (m, t) in enumerate(zip(matches, totals), start=1):
        if n > 1:
            m, t = m + 1, t + 1
        if m == 0 or t == 0:
            return 0.0
        log_precision += math.log(m / t)
    brevity = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return brevity * math.exp(log_precision / max_n)


def _prf(overlap: int, hyp_total: int, ref_total: int) -> dict[str, float]:
    precision = overlap / hyp_total if hyp_total else 0.0
    recall = overlap / ref_total if ref_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def rouge_n(reference: str, hypothesis: str, n: int) -> dict[str, float]:
    """ROUGE-N precision, recall and F1 from clipped n-gram overlap."""
    ref, hyp = _ngrams(tokenize(reference), n), _ngrams(tokenize(hypothesis), n)
    overlap = sum(min(c, ref[g]) for g, c in hyp.items())
    return _prf(overlap, sum(hyp.values()), sum(ref.values()))


def rouge_l(reference: str, hypothesis: str) -> dict[str, float]:
    """ROUGE-L precision, recall and F1 (LCS-based; f1 equals rouge_l_f1)."""
    ref, hyp = tokenize(reference), tokenize(hypothesis)
    if not ref or not hyp:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return _prf(_lcs_length(ref, hyp), len(hyp), len(ref))


def _char_ngrams(text: str, n: int) -> Counter:
    chars = "".join(text.split())  # chrF ignores whitespace
    return Counter(chars[i : i + n] for i in range(len(chars) - n + 1))


def corpus_chrf(references: Sequence[str], hypotheses: Sequence[str], max_n: int = 6, beta: float = 2.0) -> float:
    """Corpus chrF (Popović 2015): character n-gram F-beta (beta=2, n=1..6), in [0, 1].

    Matches and counts are summed over the corpus per order, precision and recall are
    averaged over the orders that occur, then combined with recall weighted beta times.
    """
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    matches = [0] * max_n
    hyp_totals = [0] * max_n
    ref_totals = [0] * max_n
    for reference, hypothesis in zip(references, hypotheses):
        for n in range(1, max_n + 1):
            ref, hyp = _char_ngrams(reference, n), _char_ngrams(hypothesis, n)
            matches[n - 1] += sum(min(c, ref[g]) for g, c in hyp.items())
            hyp_totals[n - 1] += sum(hyp.values())
            ref_totals[n - 1] += sum(ref.values())
    orders = [i for i in range(max_n) if hyp_totals[i] and ref_totals[i]]
    if not orders:
        return 0.0
    precision = sum(matches[i] / hyp_totals[i] for i in orders) / len(orders)
    recall = sum(matches[i] / ref_totals[i] for i in orders) / len(orders)
    if precision + recall == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def exact_match(reference: str, hypothesis: str) -> bool:
    """Equal after the shared lowercase word tokenization."""
    return tokenize(reference) == tokenize(hypothesis)


def copy_fraction(code: str, hypothesis: str) -> float:
    """Share of the input's distinct word tokens that reappear in the output."""
    source = set(tokenize(code))
    return len(source & set(tokenize(hypothesis))) / len(source) if source else 0.0


def meteor_available() -> tuple[bool, str | None]:
    """METEOR needs nltk and its WordNet data installed locally; nothing is downloaded."""
    try:
        from nltk.corpus import wordnet
        from nltk.translate.meteor_score import meteor_score  # noqa: F401

        wordnet.ensure_loaded()
        return True, None
    except ImportError:
        return False, "nltk is not installed"
    except LookupError:
        return False, "nltk WordNet data is not installed locally (no download at evaluation time)"


def mean_meteor(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    from nltk.translate.meteor_score import meteor_score

    return sum(
        meteor_score([tokenize(r)], tokenize(h)) for r, h in zip(references, hypotheses)
    ) / len(references)


def _percentile_ci(values, alpha: float = 0.05) -> dict[str, float]:
    import numpy as np

    low, high = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"low": float(low), "high": float(high)}


def bootstrap_bleu_rouge(
    references: Sequence[str], hypotheses: Sequence[str], n_bootstrap: int = 1000, seed: int = 42
) -> dict:
    """95% percentile CIs for corpus BLEU-4 and mean ROUGE-L F1 (shared row resamples)."""
    import numpy as np

    stats = [_bleu_stats(r, h) for r, h in zip(references, hypotheses)]
    matches = np.array([s[0] for s in stats])
    totals = np.array([s[1] for s in stats])
    ref_len = np.array([s[2] for s in stats])
    hyp_len = np.array([s[3] for s in stats])
    rouge = np.array([rouge_l_f1(r, h) for r, h in zip(references, hypotheses)])
    rng = np.random.default_rng(seed)
    n = len(references)
    bleu_values, rouge_values = [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        bleu_values.append(
            _bleu_from_stats(
                matches[idx].sum(0).tolist(), totals[idx].sum(0).tolist(), int(ref_len[idx].sum()), int(hyp_len[idx].sum())
            )
        )
        rouge_values.append(float(rouge[idx].mean()))
    return {
        "method": "percentile bootstrap",
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "bleu4": _percentile_ci(bleu_values),
        "rouge_l_f1": _percentile_ci(rouge_values),
    }


def summary_metrics(
    references: Sequence[str],
    hypotheses: Sequence[str],
    codes: Sequence[str] | None = None,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
    copy_threshold: float = 0.8,
) -> dict:
    """Every summarization metric for one system on one set of rows. Scores are in [0, 1]."""
    if len(references) != len(hypotheses) or (codes is not None and len(codes) != len(references)):
        raise ValueError("references, hypotheses and codes must have the same length")
    if not references:
        raise ValueError("no examples to score")
    n = len(references)

    def mean_prf(scores):
        return {key: sum(s[key] for s in scores) / n for key in ("precision", "recall", "f1")}

    hyp_lengths = [len(tokenize(h)) for h in hypotheses]
    ref_lengths = [len(tokenize(r)) for r in references]
    result = {
        "examples": n,
        "bleu4": corpus_bleu(references, hypotheses),
        "sentence_bleu4_smoothed": sum(sentence_bleu(r, h) for r, h in zip(references, hypotheses)) / n,
        "rouge_1": mean_prf([rouge_n(r, h, 1) for r, h in zip(references, hypotheses)]),
        "rouge_2": mean_prf([rouge_n(r, h, 2) for r, h in zip(references, hypotheses)]),
        "rouge_l": mean_prf([rouge_l(r, h) for r, h in zip(references, hypotheses)]),
        "chrf": corpus_chrf(references, hypotheses),
        "meteor": None,
        "meteor_note": None,
        "exact_match": sum(exact_match(r, h) for r, h in zip(references, hypotheses)) / n,
        "avg_generated_tokens": sum(hyp_lengths) / n,
        "avg_reference_tokens": sum(ref_lengths) / n,
        "empty_output_rate": sum(length == 0 for length in hyp_lengths) / n,
        "copy_rate": None,
        "copy_definition": f"share of outputs containing > {copy_threshold:.0%} of the input's distinct word tokens",
    }
    available, note = meteor_available()
    if available:
        result["meteor"] = mean_meteor(references, hypotheses)
    else:
        result["meteor_note"] = f"METEOR skipped: {note}"
    if codes is not None:
        result["copy_rate"] = sum(copy_fraction(c, h) > copy_threshold for c, h in zip(codes, hypotheses)) / n
    result["bootstrap"] = bootstrap_bleu_rouge(references, hypotheses, n_bootstrap, seed) if n_bootstrap else None
    return result
