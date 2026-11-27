"""Classification metrics for BUGGY (1) vs CLEAN (0) predictions.

Everything here is pure numpy/scikit-learn/scipy; nothing loads a model. Thresholds are
chosen by callers on the validation split only (see best_f1_threshold); this module never
looks at which split it is given.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

SEED = 42
N_BOOTSTRAP = 1000
CI_METRICS = ("accuracy", "bug_f1", "mcc", "roc_auc", "pr_auc")
MUTATION_TYPES = ("comparison_operator", "arithmetic_operator", "boolean_operator")


def _arrays(y_true, y_pred, y_prob=None):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.shape != y_pred.shape or y_true.ndim != 1 or not len(y_true):
        raise ValueError("y_true and y_pred must be non-empty 1-D arrays of equal length")
    if not set(np.unique(np.concatenate([y_true, y_pred]))) <= {0, 1}:
        raise ValueError("labels must be 0 (CLEAN) or 1 (BUGGY)")
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        if y_prob.shape != y_true.shape:
            raise ValueError("y_prob must match y_true")
        if np.any((y_prob < 0) | (y_prob > 1)) or np.any(np.isnan(y_prob)):
            raise ValueError("y_prob must be probabilities of BUGGY in [0, 1]")
    return y_true, y_pred, y_prob


def _safe(value):
    """JSON-friendly float: None for nan/inf."""
    value = float(value)
    return None if math.isnan(value) or math.isinf(value) else value


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _both_classes(y_true):
    return len(np.unique(y_true)) == 2


# --- single metrics used by the bootstrap -------------------------------------------


def bug_f1(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def mcc(y_true, y_pred):
    # sklearn returns 0.0 when undefined (a constant prediction or a single class).
    return float(matthews_corrcoef(y_true, y_pred))


def roc_auc(y_true, y_prob):
    return float(roc_auc_score(y_true, y_prob)) if _both_classes(y_true) else float("nan")


def pr_auc(y_true, y_prob):
    return float(average_precision_score(y_true, y_prob)) if np.any(y_true == 1) else float("nan")


# --- calibration --------------------------------------------------------------------


def calibration(y_true, y_prob, bins=10):
    """Brier score, log loss and Expected Calibration Error with equal-width bins.

    ECE = sum over bins of (bin size / n) * |mean predicted probability - observed BUGGY rate|.
    The bins are returned for a reliability diagram (empty bins have count 0 and nulls).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    # Bin i holds [edge_i, edge_i+1); the last bin also holds 1.0.
    index = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, bins - 1)
    curve, ece = [], 0.0
    for i in range(bins):
        mask = index == i
        count = int(mask.sum())
        if count:
            confidence = float(y_prob[mask].mean())
            observed = float(y_true[mask].mean())
            ece += count / len(y_true) * abs(confidence - observed)
        else:
            confidence = observed = None
        curve.append(
            {
                "bin_lower": float(edges[i]),
                "bin_upper": float(edges[i + 1]),
                "count": count,
                "mean_predicted": confidence,
                "observed_buggy_rate": observed,
            }
        )
    clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
    return {
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": _safe(log_loss(y_true, clipped, labels=[0, 1])),
        "ece": float(ece),
        "bins": bins,
        "reliability_curve": curve,
    }


# --- confidence intervals and thresholds --------------------------------------------


def bootstrap_ci(y_true, y_pred, y_prob, n_bootstrap=N_BOOTSTRAP, seed=SEED, alpha=0.05):
    """Percentile 95% CIs for CI_METRICS from row resamples (same indices for every metric).

    Resamples that contain a single class are skipped for ROC-AUC/PR-AUC; the number of
    valid resamples per metric is reported.
    """
    y_true, y_pred, y_prob = _arrays(y_true, y_pred, y_prob)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = {name: [] for name in CI_METRICS}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        t, p = y_true[idx], y_pred[idx]
        scores["accuracy"].append(float(np.mean(t == p)))
        scores["bug_f1"].append(bug_f1(t, p))
        scores["mcc"].append(mcc(t, p))
        if y_prob is not None and _both_classes(t):
            scores["roc_auc"].append(roc_auc(t, y_prob[idx]))
            scores["pr_auc"].append(pr_auc(t, y_prob[idx]))
    result = {}
    for name, values in scores.items():
        if not values:
            result[name] = None
            continue
        low, high = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        result[name] = {"low": float(low), "high": float(high), "resamples": len(values)}
    return {"method": "percentile bootstrap", "n_bootstrap": n_bootstrap, "seed": seed, "ci": result}


def best_f1_threshold(y_true, y_prob):
    """Threshold maximizing BUGGY F1. Call this on VALIDATION predictions only.

    Candidates are every distinct predicted probability (predict BUGGY when prob >= t), so
    the optimum is exact rather than a grid approximation. Ties prefer the value closest to 0.5.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    best = (-1.0, -math.inf, 0.5)
    for threshold in np.unique(y_prob):
        f1 = bug_f1(y_true, (y_prob >= threshold).astype(int))
        candidate = (f1, -abs(float(threshold) - 0.5), float(threshold))
        if candidate > best:
            best = candidate
    return {"threshold": best[2], "validation_bug_f1": best[0]}


def f1_threshold_curve(y_true, y_prob, thresholds=None):
    """BUGGY F1, precision and recall over a threshold grid (for the F1-vs-threshold plot)."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    grid = np.round(np.arange(0.02, 0.981, 0.02), 2) if thresholds is None else thresholds
    rows = []
    for t in grid:
        pred = (y_prob >= t).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, pred, labels=[1], zero_division=0
        )
        rows.append(
            {"threshold": float(t), "precision": float(precision[0]), "recall": float(recall[0]), "f1": float(f1[0])}
        )
    return rows


# --- the main entry point -------------------------------------------------------------


def compute_bug_metrics(
    y_true,
    y_pred,
    y_prob=None,
    mutation_types=None,
    *,
    n_bootstrap=N_BOOTSTRAP,
    seed=SEED,
    bins=10,
):
    """All classification metrics for one set of predictions.

    y_prob is the predicted probability of BUGGY. Probability-based metrics (ROC-AUC, PR-AUC,
    calibration) are None when it is missing. mutation_types (e.g. "clean",
    "comparison_operator", ...) adds per-mutation recall and the clean-row false-positive rate.
    """
    y_true, y_pred, y_prob = _arrays(y_true, y_pred, y_prob)
    n = len(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel().tolist()
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    counts = [[tn, fp], [fn, tp]]
    normalized = [
        [_ratio(row[0], sum(row)), _ratio(row[1], sum(row))] for row in counts
    ]
    positive_rate = float(np.mean(y_true))
    result = {
        "n": n,
        "positives": int(tp + fn),
        "negatives": int(tn + fp),
        "accuracy": float((tp + tn) / n),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if _both_classes(y_true) else None,
        "bug_precision": float(precision[1]),
        "bug_recall": float(recall[1]),
        "bug_f1": float(f1[1]),
        "clean_precision": float(precision[0]),
        "clean_recall": float(recall[0]),
        "clean_f1": float(f1[0]),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": _safe(np.average(f1, weights=support)) if support.sum() else None,
        "specificity": _ratio(tn, tn + fp),
        "false_positive_rate": _ratio(fp, tn + fp),
        "false_negative_rate": _ratio(fn, fn + tp),
        "mcc": mcc(y_true, y_pred),
        "cohen_kappa": _safe(cohen_kappa_score(y_true, y_pred)) if len(np.unique(np.concatenate([y_true, y_pred]))) > 1 else 0.0,
        "roc_auc": None,
        "pr_auc": None,
        "pr_auc_baseline": positive_rate,  # a random ranker's expected average precision
        "calibration": None,
        "confusion_matrix": {
            "labels": ["CLEAN", "BUGGY"],
            "rows": "actual",
            "columns": "predicted",
            "counts": counts,
            "row_normalized": normalized,
        },
    }
    if y_prob is not None:
        result["roc_auc"] = _safe(roc_auc(y_true, y_prob))
        result["pr_auc"] = _safe(pr_auc(y_true, y_prob))
        result["calibration"] = calibration(y_true, y_prob, bins)
    if mutation_types is not None:
        result["by_mutation_type"] = by_mutation_type(y_true, y_pred, mutation_types)
    result["bootstrap"] = bootstrap_ci(y_true, y_pred, y_prob, n_bootstrap, seed) if n_bootstrap else None
    return result


def by_mutation_type(y_true, y_pred, mutation_types):
    """Recall per mutation type (buggy rows) and the false-positive rate on clean rows."""
    y_true, y_pred, _ = _arrays(y_true, y_pred)
    types = np.asarray(mutation_types, dtype=object)
    if types.shape != y_true.shape:
        raise ValueError("mutation_types must match y_true")
    result = {}
    for name in sorted(set(types.tolist()) - {"clean"}):
        mask = (types == name) & (y_true == 1)
        hits = int(np.sum(y_pred[mask] == 1))
        result[name] = {"n": int(mask.sum()), "detected": hits, "recall": _ratio(hits, int(mask.sum()))}
    clean = (types == "clean") & (y_true == 0)
    flagged = int(np.sum(y_pred[clean] == 1))
    result["clean"] = {
        "n": int(clean.sum()),
        "false_positives": flagged,
        "false_positive_rate": _ratio(flagged, int(clean.sum())),
    }
    return result


def breakdown(y_true, y_pred, groups, y_prob=None, order=None):
    """Compact metrics per group (code-length quartile, truncated or not, ...)."""
    y_true, y_pred, y_prob = _arrays(y_true, y_pred, y_prob)
    groups = np.asarray(groups, dtype=object)
    names = order or sorted(set(groups.tolist()), key=str)
    result = {}
    for name in names:
        mask = groups == name
        if not mask.any():
            continue
        t, p = y_true[mask], y_pred[mask]
        metrics = compute_bug_metrics(t, p, None if y_prob is None else y_prob[mask], n_bootstrap=0)
        result[str(name)] = {
            "n": int(mask.sum()),
            "positives": metrics["positives"],
            "accuracy": metrics["accuracy"],
            "bug_precision": metrics["bug_precision"],
            "bug_recall": metrics["bug_recall"],
            "bug_f1": metrics["bug_f1"],
            "false_positive_rate": metrics["false_positive_rate"],
            "mcc": metrics["mcc"],
            "roc_auc": metrics["roc_auc"],
        }
    return result


def length_quartiles(n_tokens):
    """Label each row with its token-count quartile ("Q1 (≤a)", ...), edges from these rows."""
    n_tokens = np.asarray(n_tokens, dtype=float)
    edges = np.percentile(n_tokens, [25, 50, 75])
    index = np.searchsorted(edges, n_tokens, side="left")
    bounds = [float(n_tokens.min()), *edges.tolist(), float(n_tokens.max())]
    labels = [
        f"Q{i + 1} ({int(bounds[i]) + (1 if i else 0)}–{int(bounds[i + 1])} tokens)" for i in range(4)
    ]
    return [labels[i] for i in index], labels


def threshold_analysis(y_true, y_prob, tuned_threshold, mutation_types=None, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """Metrics at the default 0.5 and at a threshold tuned beforehand on validation."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    result = {}
    for name, threshold in (("default_0.5", 0.5), ("validation_tuned", tuned_threshold)):
        pred = (y_prob >= threshold).astype(int)
        metrics = compute_bug_metrics(y_true, pred, y_prob, mutation_types, n_bootstrap=n_bootstrap, seed=seed)
        result[name] = {"threshold": float(threshold), "metrics": metrics}
    return result


def curves(y_true, y_prob):
    """ROC and PR curve points for plotting."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return {
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr": {"precision": precision.tolist(), "recall": recall.tolist()},
    }


# --- paired model comparison ------------------------------------------------------------


def mcnemar(y_true, pred_a, pred_b):
    """McNemar's test on the rows where exactly one model is correct.

    b = A correct & B wrong, c = A wrong & B correct. Exact p is the two-sided binomial test
    of b vs c with p = 0.5; the chi-squared version uses the continuity correction
    (|b - c| - 1)^2 / (b + c).
    """
    from scipy.stats import binomtest, chi2

    y_true = np.asarray(y_true, dtype=int)
    correct_a = np.asarray(pred_a, dtype=int) == y_true
    correct_b = np.asarray(pred_b, dtype=int) == y_true
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    table = [
        [int(np.sum(correct_a & correct_b)), b],
        [c, int(np.sum(~correct_a & ~correct_b))],
    ]
    if b + c == 0:
        return {"table": table, "b": b, "c": c, "exact_p": 1.0, "chi2": 0.0, "chi2_p": 1.0}
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    return {
        "table": table,
        "table_layout": "rows: model A correct / wrong; columns: model B correct / wrong",
        "b": b,
        "c": c,
        "exact_p": float(binomtest(b, b + c, 0.5, alternative="two-sided").pvalue),
        "chi2": float(statistic),
        "chi2_p": float(chi2.sf(statistic, 1)),
    }


def paired_bootstrap_f1(y_true, pred_a, pred_b, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """BUGGY F1 difference (B - A) with a 95% percentile CI from shared row resamples."""
    y_true = np.asarray(y_true, dtype=int)
    pred_a = np.asarray(pred_a, dtype=int)
    pred_b = np.asarray(pred_b, dtype=int)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        diffs.append(bug_f1(y_true[idx], pred_b[idx]) - bug_f1(y_true[idx], pred_a[idx]))
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {
        "difference": "B - A",
        "observed": bug_f1(y_true, pred_b) - bug_f1(y_true, pred_a),
        "ci_low": float(low),
        "ci_high": float(high),
        "share_b_better": float(np.mean(np.asarray(diffs) > 0)),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }
