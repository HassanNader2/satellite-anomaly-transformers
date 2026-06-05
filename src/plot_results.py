"""
Generate publication-quality figures from Stage 6 experiment results.

Usage (from project root, venv active):
  python papers/satellite-anomaly/experiments/src/plot_results.py

Reads:
  experiments/results/<config-base>-summary.json       (mean +/- std metrics)
  experiments/results/<config-base>-seed42-scores.npz  (raw scores for ROC/PR)

Writes to:
  papers/satellite-anomaly/figures/
    metrics_comparison.pdf / .png
    roc_pr_curves.pdf / .png
    score_distributions.pdf / .png

Runs with whatever models are available — skips missing data gracefully.
"""

import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

# --- EAAI submission formatting ---
matplotlib.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          10,
    "axes.labelsize":     10,
    "axes.titlesize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "legend.framealpha":  0.85,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linewidth":     0.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "lines.linewidth":    1.8,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

_HERE    = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.abspath(os.path.join(_HERE, "..", "results"))
_FIGURES = os.path.abspath(os.path.join(_HERE, "..", "..", "figures"))
os.makedirs(_FIGURES, exist_ok=True)

# Wong colorblind-safe palette: blue, orange/gold, green
MODELS = [
    {
        "config_base": "anomaly-transformer-opssat-20260602-d64-01",
        "label":       "Anomaly Transformer",
        "short":       "AT",
        "color":       "#E69F00",
        "linestyle":   "-",
    },
    {
        "config_base": "patchtst-opssat-20260602-01",
        "label":       "PatchTST",
        "short":       "PatchTST",
        "color":       "#0072B2",
        "linestyle":   "--",
    },
    {
        "config_base": "itransformer-opssat-20260602-01",
        "label":       "iTransformer†",
        "short":       "iTransformer†",
        "color":       "#009E73",
        "linestyle":   ":",
    },
]

METRICS_BAR   = ["f1", "mcc", "accuracy", "aucroc", "aucpr"]
METRIC_LABELS = {"f1": "F1", "mcc": "MCC", "accuracy": "Accuracy",
                 "aucroc": "AUCROC", "aucpr": "AUCPR"}
ITRANS_NOTE   = "† iTransformer runs in degenerate single-variate mode on OPS-SAT-AD (1 variate token)."


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_summary(config_base):
    path = os.path.join(_RESULTS, f"{config_base}-summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _load_scores(config_base, preferred_seed=42):
    """Return (test_scores, test_labels) from .npz, or None if missing."""
    path = os.path.join(_RESULTS, f"{config_base}-seed{preferred_seed}-scores.npz")
    if not os.path.exists(path):
        candidates = sorted(glob.glob(
            os.path.join(_RESULTS, f"{config_base}-seed*-scores.npz")
        ))
        if not candidates:
            return None
        path = candidates[0]
    data = np.load(path)
    return data["test_scores"], data["test_labels"]


def _meta(config_base):
    return next(m for m in MODELS if m["config_base"] == config_base)


# ---------------------------------------------------------------------------
# Figure 1 — Metrics comparison bar chart
# ---------------------------------------------------------------------------

def plot_metrics_comparison(summaries):
    available = [(cb, s) for cb, s in summaries if s is not None]
    if not available:
        print("  [skip] metrics_comparison — no summary files found.")
        return

    n_metrics = len(METRICS_BAR)
    n_models  = len(available)
    bw        = 0.22
    x         = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(7.5, 4.0))

    for i, (cb, summary) in enumerate(available):
        meta   = _meta(cb)
        ms     = summary["metrics_summary"]
        means  = [ms[m]["mean"] for m in METRICS_BAR]
        stds   = [ms[m]["std"]  for m in METRICS_BAR]
        offset = (i - (n_models - 1) / 2) * bw

        ax.bar(
            x + offset, means, bw,
            yerr=stds,
            label=meta["label"],
            color=meta["color"],
            capsize=3,
            error_kw={"linewidth": 1.0, "ecolor": "black", "alpha": 0.7},
            alpha=0.88,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS_BAR])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper right", ncol=1)
    ax.set_title("OPS-SAT-AD — Model Comparison (mean ± std, 3 seeds)")
    fig.text(0.01, -0.02, ITRANS_NOTE, fontsize=7.5, color="#555555", ha="left")

    _save(fig, "metrics_comparison")


# ---------------------------------------------------------------------------
# Figure 2 — ROC and PR curves
# ---------------------------------------------------------------------------

def plot_roc_pr(score_data):
    available = [(cb, d) for cb, d in score_data if d is not None]
    if not available:
        print("  [skip] roc_pr_curves — no score files found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6))

    for cb, (scores, labels) in available:
        meta = _meta(cb)

        fpr, tpr, _   = roc_curve(labels, scores)
        prec, rec, _  = precision_recall_curve(labels, scores)

        axes[0].plot(fpr, tpr, color=meta["color"],
                     linestyle=meta["linestyle"], label=meta["label"])
        axes[1].plot(rec, prec, color=meta["color"],
                     linestyle=meta["linestyle"], label=meta["label"])

    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4, label="Random")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1.02])

    anomaly_rate = 0.22
    axes[1].axhline(anomaly_rate, color="k", linestyle="--", linewidth=0.8,
                    alpha=0.4, label=f"Random ({anomaly_rate:.2f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(loc="upper right")
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1.02])

    fig.tight_layout()
    fig.text(0.01, -0.03, ITRANS_NOTE, fontsize=7.5, color="#555555", ha="left")
    _save(fig, "roc_pr_curves")


# ---------------------------------------------------------------------------
# Figure 3 — Anomaly score distributions
# ---------------------------------------------------------------------------

def plot_score_distributions(score_data):
    available = [(cb, d) for cb, d in score_data if d is not None]
    if not available:
        print("  [skip] score_distributions — no score files found.")
        return

    n    = len(available)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (cb, (scores, labels)) in zip(axes, available):
        meta       = _meta(cb)
        nom_scores  = scores[labels == 0]
        anom_scores = scores[labels == 1]
        bins        = np.linspace(scores.min(), scores.max(), 45)

        ax.hist(nom_scores,  bins=bins, alpha=0.65, color="#888888",
                label="Nominal",   density=True)
        ax.hist(anom_scores, bins=bins, alpha=0.70, color=meta["color"],
                label="Anomalous", density=True)

        ax.set_title(meta["label"], fontsize=9)
        ax.set_xlabel("Anomaly Score")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Density")
    fig.suptitle("OPS-SAT-AD — Anomaly Score Distributions (seed 42)", y=1.02)
    fig.tight_layout()
    _save(fig, "score_distributions")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _save(fig, name):
    for fmt in ("pdf", "png"):
        out = os.path.join(_FIGURES, f"{name}.{fmt}")
        fig.savefig(out)
        print(f"  Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Results : {_RESULTS}")
    print(f"Figures : {_FIGURES}")
    print()

    summaries  = [(m["config_base"], _load_summary(m["config_base"]))  for m in MODELS]
    score_data = [(m["config_base"], _load_scores(m["config_base"]))   for m in MODELS]

    n_sum    = sum(1 for _, s in summaries  if s is not None)
    n_scores = sum(1 for _, d in score_data if d is not None)
    print(f"Summary files found : {n_sum}/3")
    print(f"Score files found   : {n_scores}/3")
    print()

    print("Generating metrics_comparison...")
    plot_metrics_comparison(summaries)

    print("Generating roc_pr_curves...")
    plot_roc_pr(score_data)

    print("Generating score_distributions...")
    plot_score_distributions(score_data)

    print("\nDone.")


if __name__ == "__main__":
    main()
