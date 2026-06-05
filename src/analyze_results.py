"""
Stage 9 — Results analysis and figure generation.

Loads all result JSONs, prints formatted tables for the paper,
and generates figures for secondary validation.

Usage:
  source venv/bin/activate
  cd papers/satellite-anomaly/experiments/src
  python analyze_results.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_HERE        = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS = os.path.abspath(os.path.join(_HERE, ".."))
_RESULTS     = os.path.join(_EXPERIMENTS, "results")
_FIGURES     = os.path.abspath(os.path.join(_EXPERIMENTS, "..", "figures"))

os.makedirs(_FIGURES, exist_ok=True)

# Wong colorblind-safe palette
COLORS = {
    "patchtst":            "#0072B2",   # blue
    "itransformer":        "#009E73",   # bluish green
    "anomaly-transformer": "#D55E00",   # vermillion
}
MODEL_LABELS = {
    "patchtst":            "PatchTST",
    "itransformer":        "iTransformer",
    "anomaly-transformer": "Anomaly Transformer",
}

# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def _load(run_id):
    path = os.path.join(_RESULTS, f"{run_id}.json")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# OPS-SAT-AD summary table
# ---------------------------------------------------------------------------

def print_opssat_table():
    models = ["anomaly-transformer", "patchtst", "itransformer"]
    summary_ids = {
        "anomaly-transformer": "anomaly-transformer-opssat-20260602-d64-01-summary",
        "patchtst":            "patchtst-opssat-20260602-01-summary",
        "itransformer":        "itransformer-opssat-20260602-01-summary",
    }

    print("\n" + "="*70)
    print("OPS-SAT-AD RESULTS (mean ± std, 3 seeds: 42/0/1)")
    print("="*70)
    header = f"{'Model':<22} {'Acc':>10} {'Prec':>10} {'Rec':>10} {'F1':>10} {'MCC':>10} {'AUCROC':>10} {'AUCPR':>10}"
    print(header)
    print("-"*70)

    for m in models:
        d = _load(summary_ids[m])
        s = d["metrics_summary"]
        acc   = f"{s['accuracy']['mean']:.3f}±{s['accuracy']['std']:.3f}"
        prec  = f"{s['precision']['mean']:.3f}±{s['precision']['std']:.3f}"
        rec   = f"{s['recall']['mean']:.3f}±{s['recall']['std']:.3f}"
        f1    = f"{s['f1']['mean']:.3f}±{s['f1']['std']:.3f}"
        mcc   = f"{s['mcc']['mean']:.3f}±{s['mcc']['std']:.3f}"
        auc   = f"{s['aucroc']['mean']:.3f}±{s['aucroc']['std']:.3f}"
        aucpr = f"{s['aucpr']['mean']:.3f}±{s['aucpr']['std']:.3f}"
        print(f"{MODEL_LABELS[m]:<22} {acc:>10} {prec:>10} {rec:>10} {f1:>10} {mcc:>10} {auc:>10} {aucpr:>10}")


# ---------------------------------------------------------------------------
# Secondary validation table
# ---------------------------------------------------------------------------

def print_secondary_table():
    print("\n" + "="*70)
    print("SECONDARY VALIDATION (seed 42, no point-adjustment)")
    print("="*70)

    for dataset in ["smap", "msl"]:
        enc_in = 25 if dataset == "smap" else 55
        anom_rate = 0.159 if dataset == "smap" else 0.127
        print(f"\n{dataset.upper()}  (enc_in={enc_in}, anomaly_rate={anom_rate:.3f})")
        print(f"{'Model':<22} {'AUCROC':>10} {'AUCPR':>10} {'F1(oracle)':>12} {'Note':>20}")
        print("-"*70)

        degenerate_f1 = 2*anom_rate/(1+anom_rate)

        for m in ["anomaly-transformer", "patchtst", "itransformer"]:
            run_id = f"{m}-{dataset}-20260605-01-seed42"
            d = _load(run_id)
            me = d["metrics"]
            is_degen = abs(me["f1_oracle"] - degenerate_f1) < 0.002
            note = "degenerate" if is_degen else ""
            print(f"{MODEL_LABELS[m]:<22} {me['aucroc']:>10.4f} {me['aucpr']:>10.4f} {me['f1_oracle']:>12.4f} {note:>20}")


# ---------------------------------------------------------------------------
# Secondary validation figure — AUCROC grouped bar chart
# ---------------------------------------------------------------------------

def plot_secondary_aucroc():
    models = ["patchtst", "itransformer", "anomaly-transformer"]
    datasets = ["smap", "msl"]
    dataset_labels = {"smap": "SMAP", "msl": "MSL"}

    aucroc = {}
    for m in models:
        aucroc[m] = {}
        for ds in datasets:
            run_id = f"{m}-{ds}-20260605-01-seed42"
            d = _load(run_id)
            aucroc[m][ds] = d["metrics"]["aucroc"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)

    x = np.arange(len(models))
    width = 0.55

    for ax, ds in zip(axes, datasets):
        bars = [aucroc[m][ds] for m in models]
        colors = [COLORS[m] for m in models]
        rects = ax.bar(x, bars, width, color=colors, edgecolor="white", linewidth=0.8, zorder=3)

        # Random baseline
        ax.axhline(0.5, color="black", linewidth=1.2, linestyle="--", label="Random (0.5)", zorder=2)

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=9)
        ax.set_ylabel("AUCROC", fontsize=10)
        ax.set_title(dataset_labels[ds], fontsize=11, fontweight="bold")
        ax.set_ylim(0.0, 0.75)
        ax.yaxis.grid(True, linestyle=":", alpha=0.6, zorder=0)
        ax.set_axisbelow(True)

        for rect, val in zip(rects, bars):
            ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.012,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8.5)

    # Shared legend
    legend_handles = [
        mpatches.Patch(color=COLORS[m], label=MODEL_LABELS[m]) for m in models
    ] + [plt.Line2D([0], [0], color="black", linewidth=1.2, linestyle="--", label="Random (0.5)")]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4,
               fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("Secondary Validation — AUCROC without Point-Adjustment",
                 fontsize=11, y=1.10)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = os.path.join(_FIGURES, f"secondary_validation.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-dataset AUCROC comparison figure
# ---------------------------------------------------------------------------

def plot_cross_dataset_aucroc():
    """
    Grouped bar chart: 3 models × 3 datasets (OPS-SAT-AD, SMAP, MSL).
    OPS-SAT-AD uses mean across 3 seeds; SMAP/MSL use seed 42.
    """
    models   = ["patchtst", "itransformer", "anomaly-transformer"]
    datasets = ["opssat", "smap", "msl"]
    dataset_labels = {"opssat": "OPS-SAT-AD", "smap": "SMAP", "msl": "MSL"}

    summary_ids = {
        "patchtst":            "patchtst-opssat-20260602-01-summary",
        "itransformer":        "itransformer-opssat-20260602-01-summary",
        "anomaly-transformer": "anomaly-transformer-opssat-20260602-d64-01-summary",
    }

    aucroc = {m: {} for m in models}
    aucroc_err = {m: {} for m in models}

    for m in models:
        # OPS-SAT-AD — mean ± std from summary
        d = _load(summary_ids[m])
        aucroc[m]["opssat"]     = d["metrics_summary"]["aucroc"]["mean"]
        aucroc_err[m]["opssat"] = d["metrics_summary"]["aucroc"]["std"]
        # SMAP / MSL — point estimate, no std
        for ds in ("smap", "msl"):
            run_id = f"{m}-{ds}-20260605-01-seed42"
            d2 = _load(run_id)
            aucroc[m][ds]     = d2["metrics"]["aucroc"]
            aucroc_err[m][ds] = 0.0

    n_datasets = len(datasets)
    n_models   = len(models)
    width      = 0.22
    x          = np.arange(n_datasets)

    fig, ax = plt.subplots(figsize=(9, 5))

    offsets = np.linspace(-(n_models-1)/2*width, (n_models-1)/2*width, n_models)

    for i, m in enumerate(models):
        vals  = [aucroc[m][ds] for ds in datasets]
        errs  = [aucroc_err[m][ds] for ds in datasets]
        pos   = x + offsets[i]
        ax.bar(pos, vals, width,
               color=COLORS[m], label=MODEL_LABELS[m],
               edgecolor="white", linewidth=0.8, zorder=3,
               yerr=errs, capsize=3, error_kw={"elinewidth": 1.0, "ecolor": "black"})

    ax.axhline(0.5, color="black", linewidth=1.2, linestyle="--",
               label="Random (0.5)", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels([dataset_labels[ds] for ds in datasets], fontsize=11)
    ax.set_ylabel("AUCROC", fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.grid(True, linestyle=":", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("AUCROC Across Datasets (no point-adjustment)", fontsize=12)

    ax.legend(fontsize=9, frameon=True, loc="upper right")
    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = os.path.join(_FIGURES, f"cross_dataset_aucroc.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# LaTeX table helpers
# ---------------------------------------------------------------------------

def print_latex_opssat():
    models = ["patchtst", "itransformer", "anomaly-transformer"]
    summary_ids = {
        "anomaly-transformer": "anomaly-transformer-opssat-20260602-d64-01-summary",
        "patchtst":            "patchtst-opssat-20260602-01-summary",
        "itransformer":        "itransformer-opssat-20260602-01-summary",
    }

    print("\n--- LaTeX OPS-SAT-AD table ---")
    print(r"\begin{tabular}{lrrrrrrr}")
    print(r"\toprule")
    print(r"Model & Acc & Prec & Rec & F1 & MCC & AUCROC & AUCPR \\")
    print(r"\midrule")
    for m in ["itransformer", "patchtst", "anomaly-transformer"]:
        d = _load(summary_ids[m])
        s = d["metrics_summary"]
        def fmt(k):
            return f"${s[k]['mean']:.3f}\\pm{s[k]['std']:.3f}$"
        row = " & ".join([
            MODEL_LABELS[m],
            fmt("accuracy"), fmt("precision"), fmt("recall"),
            fmt("f1"), fmt("mcc"), fmt("aucroc"), fmt("aucpr"),
        ])
        print(row + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def print_latex_secondary():
    models = ["itransformer", "patchtst", "anomaly-transformer"]

    print("\n--- LaTeX Secondary Validation table ---")
    print(r"\begin{tabular}{llrrr}")
    print(r"\toprule")
    print(r"Dataset & Model & AUCROC & AUCPR & F1\textsuperscript{oracle} \\")
    print(r"\midrule")
    for ds, label in [("smap", "SMAP"), ("msl", "MSL")]:
        first = True
        for m in models:
            run_id = f"{m}-{ds}-20260605-01-seed42"
            d = _load(run_id)
            me = d["metrics"]
            ds_col = label if first else ""
            first = False
            print(f"{ds_col} & {MODEL_LABELS[m]} & {me['aucroc']:.4f} & {me['aucpr']:.4f} & {me['f1_oracle']:.4f} \\\\")
        print(r"\midrule")
    print(r"\bottomrule")
    print(r"\end{tabular}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_opssat_table()
    print_secondary_table()
    print_latex_opssat()
    print_latex_secondary()
    plot_secondary_aucroc()
    plot_cross_dataset_aucroc()
    print("\nStage 9 complete.")
