"""
Stage 9 / 20260625 — Results analysis and figure generation.

Loads the from-scratch 20260625 summaries, prints tables, and generates figures:
  - secondary_validation  (3 transformers x {SMAP,MSL}, 3-seed mean +/- std)
  - cross_dataset_aucroc  (3 transformers x {OPS-SAT,SMAP,MSL}, 3-seed mean +/- std)
  - capacity_ablation     (4 models x {64,128,256,512}, AUCROC mean +/- std)

Headline capacity is d=512. per_channel_f1 is produced by analyze_per_channel.py;
metrics_comparison / roc_pr_curves / score_distributions by plot_results.py.

Usage (from project root, venv active):
  python papers/satellite-anomaly/experiments/src/analyze_results.py
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

COLORS = {
    "patchtst":            "#0072B2",   # blue
    "itransformer":        "#009E73",   # bluish green
    "anomaly-transformer": "#D55E00",   # vermillion
    "lstm-ae":             "#CC79A7",   # reddish purple
}
MODEL_LABELS = {
    "patchtst":            "PatchTST",
    "itransformer":        "iTransformer",
    "anomaly-transformer": "Anomaly Transformer",
    "lstm-ae":             "LSTM-AE",
}

# Headline d=512 summaries (3-seed).
OPSSAT_D512 = {m: f"{m}-opssat-20260625-d512-summary" for m in MODEL_LABELS}
# Full capacity sweep.
CAPS = [64, 128, 256, 512]
ABLATION_MODELS = ["itransformer", "patchtst", "lstm-ae", "anomaly-transformer"]
SECONDARY_MODELS = ["patchtst", "itransformer", "anomaly-transformer"]


def _load(run_id):
    with open(os.path.join(_RESULTS, f"{run_id}.json")) as f:
        return json.load(f)


def _load_summary(base):
    path = os.path.join(_RESULTS, f"{base}-summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def print_opssat_table():
    print("\n" + "=" * 78)
    print("OPS-SAT-AD RESULTS @ d=512 (mean +/- std, 3 seeds: 42/0/1)")
    print("=" * 78)
    print(f"{'Model':<22} {'Acc':>11} {'F1':>11} {'MCC':>11} {'AUCROC':>11} {'AUCPR':>11}")
    print("-" * 78)
    for m in ABLATION_MODELS:
        d = _load_summary(OPSSAT_D512[m].replace("-summary", ""))
        if d is None:
            continue
        s = d["metrics_summary"]
        def f(k): return f"{s[k]['mean']:.3f}±{s[k]['std']:.3f}"
        print(f"{MODEL_LABELS[m]:<22} {f('accuracy'):>11} {f('f1'):>11} {f('mcc'):>11} {f('aucroc'):>11} {f('aucpr'):>11}")


def print_secondary_table():
    print("\n" + "=" * 78)
    print("SECONDARY VALIDATION @ d=512 (3-seed mean +/- std, no point-adjustment)")
    print("=" * 78)
    for ds in ["smap", "msl"]:
        print(f"\n{ds.upper()}")
        print(f"{'Model':<22} {'AUCROC':>16} {'AUCPR':>16} {'F1(oracle)':>16}")
        print("-" * 72)
        for m in SECONDARY_MODELS:
            d = _load_summary(f"{m}-{ds}-20260625")
            if d is None:
                continue
            s = d["metrics_summary"]
            def f(k): return f"{s[k]['mean']:.3f}±{s[k]['std']:.3f}"
            print(f"{MODEL_LABELS[m]:<22} {f('aucroc'):>16} {f('aucpr'):>16} {f('f1_oracle'):>16}")


# ---------------------------------------------------------------------------
# Secondary AUCROC figure
# ---------------------------------------------------------------------------

def plot_secondary_aucroc():
    datasets = ["smap", "msl"]
    dataset_labels = {"smap": "SMAP", "msl": "MSL"}

    vals, errs = {}, {}
    for m in SECONDARY_MODELS:
        vals[m], errs[m] = {}, {}
        for ds in datasets:
            d = _load_summary(f"{m}-{ds}-20260625")
            vals[m][ds] = d["metrics_summary"]["aucroc"]["mean"]
            errs[m][ds] = d["metrics_summary"]["aucroc"]["std"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    x = np.arange(len(SECONDARY_MODELS))
    for ax, ds in zip(axes, datasets):
        bars = [vals[m][ds] for m in SECONDARY_MODELS]
        e    = [errs[m][ds] for m in SECONDARY_MODELS]
        colors = [COLORS[m] for m in SECONDARY_MODELS]
        rects = ax.bar(x, bars, 0.55, color=colors, edgecolor="white", linewidth=0.8,
                       yerr=e, capsize=3, error_kw={"elinewidth": 1.0, "ecolor": "black"}, zorder=3)
        ax.axhline(0.5, color="black", linewidth=1.2, linestyle="--", zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in SECONDARY_MODELS], fontsize=8)
        ax.set_ylabel("AUCROC")
        ax.set_title(dataset_labels[ds], fontweight="bold")
        ax.set_ylim(0.0, 0.75)
        ax.yaxis.grid(True, linestyle=":", alpha=0.6, zorder=0)
        ax.set_axisbelow(True)
        for rect, val in zip(rects, bars):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.012,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8.5)

    handles = [mpatches.Patch(color=COLORS[m], label=MODEL_LABELS[m]) for m in SECONDARY_MODELS]
    handles += [plt.Line2D([0], [0], color="black", linewidth=1.2, linestyle="--", label="Random (0.5)")]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Secondary Validation — AUCROC without Point-Adjustment (d=512, 3-seed)", y=1.10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(_FIGURES, f"secondary_validation.{ext}"), dpi=300, bbox_inches="tight")
        print(f"Saved: secondary_validation.{ext}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-dataset AUCROC figure
# ---------------------------------------------------------------------------

def plot_cross_dataset_aucroc():
    datasets = ["opssat", "smap", "msl"]
    dataset_labels = {"opssat": "OPS-SAT-AD", "smap": "SMAP", "msl": "MSL"}

    aucroc = {m: {} for m in SECONDARY_MODELS}
    err    = {m: {} for m in SECONDARY_MODELS}
    for m in SECONDARY_MODELS:
        d = _load_summary(OPSSAT_D512[m].replace("-summary", ""))
        aucroc[m]["opssat"] = d["metrics_summary"]["aucroc"]["mean"]
        err[m]["opssat"]    = d["metrics_summary"]["aucroc"]["std"]
        for ds in ("smap", "msl"):
            d2 = _load_summary(f"{m}-{ds}-20260625")
            aucroc[m][ds] = d2["metrics_summary"]["aucroc"]["mean"]
            err[m][ds]    = d2["metrics_summary"]["aucroc"]["std"]

    n_models = len(SECONDARY_MODELS)
    width = 0.22
    x = np.arange(len(datasets))
    offsets = np.linspace(-(n_models - 1) / 2 * width, (n_models - 1) / 2 * width, n_models)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(SECONDARY_MODELS):
        vals = [aucroc[m][ds] for ds in datasets]
        errs = [err[m][ds] for ds in datasets]
        ax.bar(x + offsets[i], vals, width, color=COLORS[m], label=MODEL_LABELS[m],
               edgecolor="white", linewidth=0.8, zorder=3,
               yerr=errs, capsize=3, error_kw={"elinewidth": 1.0, "ecolor": "black"})
    ax.axhline(0.5, color="black", linewidth=1.2, linestyle="--", label="Random (0.5)", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([dataset_labels[ds] for ds in datasets])
    ax.set_ylabel("AUCROC")
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.grid(True, linestyle=":", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("AUCROC Across Datasets (d=512, 3-seed, no point-adjustment)")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(_FIGURES, f"cross_dataset_aucroc.{ext}"), dpi=300, bbox_inches="tight")
        print(f"Saved: cross_dataset_aucroc.{ext}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Capacity ablation figure (NEW)
# ---------------------------------------------------------------------------

def plot_capacity_ablation():
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for m in ABLATION_MODELS:
        means, stds, xs = [], [], []
        for c in CAPS:
            d = _load_summary(f"{m}-opssat-20260625-d{c}")
            if d is None:
                continue
            means.append(d["metrics_summary"]["aucroc"]["mean"])
            stds.append(d["metrics_summary"]["aucroc"]["std"])
            xs.append(c)
        if not xs:
            continue
        ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3, linewidth=1.8,
                    color=COLORS[m], label=MODEL_LABELS[m])

    ax.set_xscale("log", base=2)
    ax.set_xticks(CAPS)
    ax.set_xticklabels([str(c) for c in CAPS])
    ax.set_xlabel("Capacity (d_model; hidden_size for LSTM-AE)")
    ax.set_ylabel("AUCROC")
    ax.set_title("OPS-SAT-AD — Capacity Ablation (3-seed mean ± std, identical training rule)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(_FIGURES, f"capacity_ablation.{ext}"), dpi=300, bbox_inches="tight")
        print(f"Saved: capacity_ablation.{ext}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------

def print_latex_opssat():
    print("\n--- LaTeX OPS-SAT-AD table (d=512) ---")
    print(r"\begin{tabular}{lrrrrr}")
    print(r"\toprule")
    print(r"Model & Acc & F1 & MCC & AUCROC & AUCPR \\")
    print(r"\midrule")
    for m in ABLATION_MODELS:
        d = _load_summary(OPSSAT_D512[m].replace("-summary", ""))
        if d is None:
            continue
        s = d["metrics_summary"]
        def fmt(k): return f"${s[k]['mean']:.3f}\\pm{s[k]['std']:.3f}$"
        print(" & ".join([MODEL_LABELS[m], fmt("accuracy"), fmt("f1"),
                          fmt("mcc"), fmt("aucroc"), fmt("aucpr")]) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def print_latex_secondary():
    print("\n--- LaTeX Secondary Validation table (d=512, 3-seed) ---")
    print(r"\begin{tabular}{llrrr}")
    print(r"\toprule")
    print(r"Dataset & Model & AUCROC & AUCPR & F1\textsuperscript{oracle} \\")
    print(r"\midrule")
    for ds, label in [("smap", "SMAP"), ("msl", "MSL")]:
        first = True
        for m in SECONDARY_MODELS:
            d = _load_summary(f"{m}-{ds}-20260625")
            if d is None:
                continue
            s = d["metrics_summary"]
            ds_col = label if first else ""
            first = False
            print(f"{ds_col} & {MODEL_LABELS[m]} & "
                  f"${s['aucroc']['mean']:.3f}\\pm{s['aucroc']['std']:.3f}$ & "
                  f"${s['aucpr']['mean']:.3f}\\pm{s['aucpr']['std']:.3f}$ & "
                  f"${s['f1_oracle']['mean']:.3f}\\pm{s['f1_oracle']['std']:.3f}$ \\\\")
        print(r"\midrule")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    print_opssat_table()
    print_secondary_table()
    print_latex_opssat()
    print_latex_secondary()
    plot_secondary_aucroc()
    plot_cross_dataset_aucroc()
    plot_capacity_ablation()
    print("\nStage 9 (20260625) complete.")
