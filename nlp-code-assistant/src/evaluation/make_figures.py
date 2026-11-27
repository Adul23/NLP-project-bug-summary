"""Render evaluation figures from saved results (no models are loaded).

    python -m src.evaluation.make_figures

Reads results/model_comparison.json, the prediction CSVs it lists and, if present,
results/summarization_metrics.json. Writes PNGs to results/figures/. A figure whose data
is missing is skipped and reported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.bug_detection.split_dataset import ROOT  # noqa: E402
from src.evaluation import bug_metrics as bm  # noqa: E402

COLORS = {"tfidf": "#0072B2", "codebert": "#D55E00"}  # Okabe-Ito, colorblind-safe
SUMMARY_COLORS = ["#999999", "#56B4E9", "#009E73", "#D55E00"]
NAMES = {"tfidf": "TF-IDF + LR", "codebert": "CodeBERT + LoRA"}
GRID = {"color": "#dddddd", "linewidth": 0.6}


def _style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, fontsize=11, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save(fig, folder, name, written):
    path = folder / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path.name)


def load_predictions(comparison, root):
    predictions = {}
    for key, block in comparison.get("evaluation", {}).items():
        files = block.get("prediction_files", {})
        frames = {split: pd.read_csv(root / path) for split, path in files.items() if (root / path).exists()}
        if {"validation", "test"} <= set(frames):
            predictions[key] = frames
    return predictions


def make_figures(results_dir=ROOT / "results", root=ROOT):
    results_dir = Path(results_dir)
    folder = results_dir / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    comparison = json.loads((results_dir / "model_comparison.json").read_text())
    predictions = load_predictions(comparison, root)
    evaluation = comparison.get("evaluation", {})
    written, skipped = [], []
    keys = [k for k in NAMES if k in predictions]

    if keys:
        fig, ax = plt.subplots(figsize=(5, 4.4))
        for k in keys:
            test = predictions[k]["test"]
            c = bm.curves(test.label, test.bug_probability)["roc"]
            ax.plot(c["fpr"], c["tpr"], color=COLORS[k], lw=1.8,
                    label=f"{NAMES[k]} (AUC {evaluation[k]['test']['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], color="#999999", lw=1, ls="--", label="Random")
        _style(ax, "ROC curve (test)", "False positive rate", "True positive rate")
        ax.legend(frameon=False, loc="lower right")
        _save(fig, folder, "roc_curves.png", written)

        fig, ax = plt.subplots(figsize=(5, 4.4))
        for k in keys:
            test = predictions[k]["test"]
            c = bm.curves(test.label, test.bug_probability)["pr"]
            ax.plot(c["recall"], c["precision"], color=COLORS[k], lw=1.8,
                    label=f"{NAMES[k]} (AP {evaluation[k]['test']['pr_auc']:.3f})")
        base = evaluation[keys[0]]["test"]["pr_auc_baseline"]
        ax.axhline(base, color="#999999", lw=1, ls="--", label=f"Random (positive rate {base:.2f})")
        _style(ax, "Precision-recall curve (test, BUGGY)", "Recall", "Precision")
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, loc="upper right")
        _save(fig, folder, "pr_curves.png", written)

        fig, ax = plt.subplots(figsize=(5, 4.4))
        ax.plot([0, 1], [0, 1], color="#999999", lw=1, ls="--", label="Perfect calibration")
        for k in keys:
            cal = evaluation[k]["test"]["calibration"]
            points = [b for b in cal["reliability_curve"] if b["count"]]
            ax.plot([b["mean_predicted"] for b in points], [b["observed_buggy_rate"] for b in points],
                    marker="o", ms=4, color=COLORS[k], lw=1.6, label=f"{NAMES[k]} (ECE {cal['ece']:.3f})")
        _style(ax, "Reliability diagram (test, 10 bins)", "Mean predicted P(BUGGY)", "Observed BUGGY rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, loc="upper left")
        _save(fig, folder, "reliability.png", written)

        fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.8), squeeze=False)
        for ax, k in zip(axes[0], keys):
            cm = evaluation[k]["test"]["confusion_matrix"]
            normalized = np.array(cm["row_normalized"], dtype=float)
            ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{cm['counts'][i][j]}\n{normalized[i, j]:.1%}", ha="center", va="center",
                            color="white" if normalized[i, j] > 0.55 else "#222222", fontsize=10)
            ax.set_xticks([0, 1], ["CLEAN", "BUGGY"])
            ax.set_yticks([0, 1], ["CLEAN", "BUGGY"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"{NAMES[k]} (test, threshold 0.5)", fontsize=11, loc="left")
        _save(fig, folder, "confusion_matrices.png", written)

        types = list(bm.MUTATION_TYPES)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        width = 0.8 / len(keys)
        for i, k in enumerate(keys):
            block = evaluation[k]["test"]["by_mutation_type"]
            values = [block[t]["recall"] for t in types] + [block["clean"]["false_positive_rate"]]
            xs = np.arange(len(values)) + (i - (len(keys) - 1) / 2) * width
            bars = ax.bar(xs, values, width, color=COLORS[k], label=NAMES[k])
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
        ax.set_xticks(np.arange(len(types) + 1),
                      [t.replace("_operator", "") for t in types] + ["clean (FPR)"])
        ax.set_ylim(0, 1.08)
        _style(ax, "Recall per mutation type, false-positive rate on clean rows (test)", None, "Rate")
        ax.legend(frameon=False, loc="upper right", fontsize=8)
        _save(fig, folder, "mutation_recall.png", written)

        fig, ax = plt.subplots(figsize=(6, 4))
        for k in keys:
            for split, style in (("validation", "-"), ("test", ":")):
                frame = predictions[k][split]
                curve = bm.f1_threshold_curve(frame.label, frame.bug_probability)
                ax.plot([r["threshold"] for r in curve], [r["f1"] for r in curve], ls=style, lw=1.6,
                        color=COLORS[k], label=f"{NAMES[k]} ({split})")
            tuned = evaluation[k]["threshold"]["validation_tuned_threshold"]
            ax.axvline(tuned, color=COLORS[k], lw=0.9, ls="--")
            ax.annotate(f"tuned on val: {tuned:.2f}", (tuned, 0.04), color=COLORS[k], fontsize=8,
                        rotation=90, ha="right", va="bottom")
        _style(ax, "BUGGY F1 vs decision threshold", "Threshold", "BUGGY F1")
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        _save(fig, folder, "f1_vs_threshold.png", written)
    else:
        skipped.append("ROC/PR/reliability/confusion/mutation/threshold figures: no prediction CSVs")

    measured = [r for r in comparison["models"] if r.get("status") == "measured" and r.get("efficiency")]
    if measured:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        labels = [r["model"] for r in measured]
        p50 = [r["efficiency"]["cpu_batch1"]["latency_ms_p50"] for r in measured]
        p95 = [r["efficiency"]["cpu_batch1"]["latency_ms_p95"] for r in measured]
        y = np.arange(len(labels))
        colors = [COLORS["tfidf" if "TF-IDF" in name else "codebert"] for name in labels]
        ax.barh(y + 0.18, p50, 0.34, color=colors, label="p50")
        ax.barh(y - 0.18, p95, 0.34, color=colors, alpha=0.45, label="p95")
        for yy, a, b in zip(y, p50, p95):
            ax.text(a * 1.08, yy + 0.18, f"{a:.1f} ms", va="center", fontsize=8)
            ax.text(b * 1.08, yy - 0.18, f"{b:.1f} ms", va="center", fontsize=8)
        ax.set_xscale("log")
        ax.set_yticks(y, labels)
        _style(ax, "CPU latency per function, batch 1 (log scale)", "Milliseconds", None)
        ax.legend(frameon=False, loc="lower right", fontsize=8)
        _save(fig, folder, "latency.png", written)

    summary_path = results_dir / "summarization_metrics.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        systems = summary["systems"]
        metrics = [("BLEU-4", lambda s: s["bleu4"]), ("ROUGE-1", lambda s: s["rouge_1"]["f1"]),
                   ("ROUGE-2", lambda s: s["rouge_2"]["f1"]), ("ROUGE-L", lambda s: s["rouge_l"]["f1"]),
                   ("chrF", lambda s: s["chrf"])]
        if all(s.get("meteor") is not None for s in systems.values()):
            metrics.append(("METEOR", lambda s: s["meteor"]))
        fig, ax = plt.subplots(figsize=(7, 3.8))
        width = 0.8 / len(systems)
        for i, (key, system) in enumerate(systems.items()):
            xs = np.arange(len(metrics)) + (i - (len(systems) - 1) / 2) * width
            ax.bar(xs, [f(system) for _, f in metrics], width, label=system["name"],
                   color=SUMMARY_COLORS[i % len(SUMMARY_COLORS)])
        ax.set_xticks(np.arange(len(metrics)), [m for m, _ in metrics])
        _style(ax, f"Summarization metrics ({summary['split']}, {summary['examples']} rows)", None, "Score (0-1)")
        ax.legend(frameon=False, fontsize=8)
        _save(fig, folder, "summarization_metrics.png", written)
    else:
        skipped.append("summarization_metrics.png: results/summarization_metrics.json not found")
    return written, skipped


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    a = p.parse_args()
    written, skipped = make_figures(a.results_dir)
    print("written:", ", ".join(written) or "none")
    for item in skipped:
        print("skipped:", item)


if __name__ == "__main__":
    main()
