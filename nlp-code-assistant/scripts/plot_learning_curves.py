"""Plot learning curves from results/learning_curves.json.

    python scripts/plot_learning_curves.py
writes results/figures/codebert_learning_curves.png and codet5_val_loss.png.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/learning_curves.json"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results/figures"

SURFACE, INK, MUTED, GRID = "#faf8f2", "#14151a", "#4a4a52", "#e6dfd2"
SERIES = ["#257de0", "#f47230", "#00b894"]  # fixed order, validated on SURFACE

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 1, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
})


def label_end(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(18, 0), textcoords="offset points",
                va="center", fontsize=11, color=INK)


def codebert(data):
    seeds = data["codebert_lora"]["seeds"]
    fig, (loss_ax, f1_ax) = plt.subplots(1, 2, figsize=(13, 5))
    for color, (seed, run) in zip(SERIES, seeds.items()):
        tr, va = run["train_loss"], run["validation"]
        loss_ax.plot(tr["epoch"], tr["loss"], color=color, lw=2, ls="--", alpha=0.85)
        loss_ax.plot(va["epoch"], va["loss"], color=color, lw=2, marker="o", ms=7,
                     markeredgecolor=SURFACE, markeredgewidth=2, label=f"seed {seed}")
        f1_ax.plot(va["epoch"], va["bug_f1"], color=color, lw=2, marker="o", ms=7,
                   markeredgecolor=SURFACE, markeredgewidth=2, label=f"seed {seed}")
        best = run["selected_epoch"]
        f1_ax.plot(best, va["bug_f1"][best - 1], marker="o", ms=16, mfc="none",
                   mec=INK, mew=1.8, zorder=5)
        label_end(f1_ax, va["epoch"][-1], va["bug_f1"][-1], f"seed {seed}")
    seed_handles = [Line2D([], [], color=c, lw=2, marker="o", ms=7, label=f"seed {s}")
                    for c, s in zip(SERIES, seeds)]
    style_handles = [Line2D([], [], color=MUTED, lw=2, ls="--", label="train (every 500 steps)"),
                     Line2D([], [], color=MUTED, lw=2, marker="o", ms=7, label="validation (per epoch)")]
    best_handle = Line2D([], [], marker="o", ms=14, mfc="none", mec=INK, ls="none",
                         label="selected checkpoint")
    loss_ax.set(title="Cross-entropy loss", xlabel="epoch", ylabel="loss", xlim=(0.5, 5.5))
    f1_ax.set(title="Validation F1 (BUGGY class)", xlabel="epoch", ylabel="F1",
              xlim=(0.7, 6.1), xticks=[1, 2, 3, 4, 5])
    loss_ax.legend(handles=seed_handles + style_handles, frameon=False, fontsize=10)
    f1_ax.legend(handles=seed_handles + [best_handle], frameon=False, fontsize=10, loc="lower right")
    fig.suptitle("CodeBERT + LoRA: learning curves, 3 seeds (Colab T4, batch 16, fp16)",
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "codebert_learning_curves.png", dpi=200)


def codet5(data):
    runs = data["codet5_small"]["runs"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for color, (name, run) in zip(SERIES, runs.items()):
        ax.plot(run["epoch"], run["val_loss"], color=color, lw=2, marker="o", ms=7,
                markeredgecolor=SURFACE, markeredgewidth=2, label=name)
        best = run["selected_epoch"]
        ax.plot(best, run["val_loss"][best - 1], marker="o", ms=16, mfc="none",
                mec=INK, mew=1.8, zorder=5)
        label_end(ax, run["epoch"][-1], run["val_loss"][-1], name)
    handles = [Line2D([], [], color=c, lw=2, marker="o", ms=7, label=n) for c, n in zip(SERIES, runs)]
    handles.append(Line2D([], [], marker="o", ms=14, mfc="none", mec=INK, ls="none",
                          label="selected checkpoint"))
    ax.set(title="CodeT5-small: validation loss per epoch (Apple M4, batch 8)",
           xlabel="epoch", ylabel="validation cross-entropy", xlim=(0.5, 13.9),
           xticks=range(1, 13))
    ax.legend(handles=handles, frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "codet5_val_loss.png", dpi=200)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads(DATA.read_text())
    codebert(data)
    codet5(data)
    print("wrote", OUT / "codebert_learning_curves.png", OUT / "codet5_val_loss.png")
