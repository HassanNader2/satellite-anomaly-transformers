"""
Phase 1.2 — Per-channel TP/FP/TN/FN breakdown on OPS-SAT-AD test set.

For each model: load the seed-42 scores NPZ and the best-F1 threshold from the
seed-42 result JSON, then compute per-channel precision/recall/F1.

Usage (from project root, venv active):
  python papers/satellite-anomaly/experiments/src/analyze_per_channel.py

Outputs:
  experiments/results/per_channel_breakdown.json
  experiments/results/per_channel_breakdown.csv
  papers/satellite-anomaly/figures/per_channel_f1.pdf
  papers/satellite-anomaly/figures/per_channel_f1.png
"""

import os
import sys
import json
import numpy as np
import pandas as pd

_HERE        = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS = os.path.abspath(os.path.join(_HERE, ".."))
_RESULTS     = os.path.join(_EXPERIMENTS, "results")
_FIGURES     = os.path.abspath(os.path.join(_EXPERIMENTS, "..", "figures"))
_DATA        = os.path.abspath(os.path.join(_EXPERIMENTS, "..", "data"))
os.makedirs(_FIGURES, exist_ok=True)

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data_loader import load_opssat

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.grid": True,
    "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

WONG = {
    "patchtst":    "#0072B2",
    "itransformer": "#009E73",
    "at":          "#E69F00",
    "lstm-ae":     "#CC79A7",
}

# Models to analyse: (display_label, scores_npz, result_json)
MODELS = [
    ("iTransformer",
     "itransformer-opssat-20260625-d512-seed42-scores.npz",
     "itransformer-opssat-20260625-d512-seed42.json",
     "itransformer"),
    ("PatchTST",
     "patchtst-opssat-20260625-d512-seed42-scores.npz",
     "patchtst-opssat-20260625-d512-seed42.json",
     "patchtst"),
    ("Anomaly Transformer",
     "anomaly-transformer-opssat-20260625-d512-seed42-scores.npz",
     "anomaly-transformer-opssat-20260625-d512-seed42.json",
     "at"),
    ("LSTM-AE",
     "lstm-ae-opssat-20260625-d512-seed42-scores.npz",
     "lstm-ae-opssat-20260625-d512-seed42.json",
     "lstm-ae"),
]

FRAGILE_CHANNELS = {"CADC0886", "CADC0890"}


def load_test_channels(csv_path, T=512, min_len=16, val_frac=0.20, seed=42):
    """
    Return the channel name for each test segment, in the same order as
    load_opssat produces the test OPSSATDataset.
    """
    df = pd.read_csv(csv_path)

    seg_table = (
        df.groupby("segment")
        .agg(
            channel=("channel", "first"),
            anomaly=("anomaly", "first"),
            is_train=("train", "first"),
            length=("value", "count"),
        )
        .reset_index()
    )
    seg_table = seg_table[seg_table["length"] >= min_len].reset_index(drop=True)

    test_df = seg_table[seg_table["is_train"] == 0].reset_index(drop=True)
    return test_df["channel"].tolist(), test_df["anomaly"].tolist()


def per_channel_stats(scores, labels, channels, threshold):
    """
    Compute TP/FP/TN/FN per channel.

    scores:    (N,) float array
    labels:    (N,) int array (1=anomalous, 0=nominal)
    channels:  (N,) list of channel name strings
    threshold: float scalar

    Returns a dict keyed by channel name.
    """
    preds = (scores >= threshold).astype(int)
    unique_channels = sorted(set(channels))
    result = {}
    for ch in unique_channels:
        idx = [i for i, c in enumerate(channels) if c == ch]
        ch_labels = np.array([labels[i] for i in idx])
        ch_preds  = np.array([preds[i] for i in idx])
        tp = int(((ch_preds == 1) & (ch_labels == 1)).sum())
        fp = int(((ch_preds == 1) & (ch_labels == 0)).sum())
        tn = int(((ch_preds == 0) & (ch_labels == 0)).sum())
        fn = int(((ch_preds == 0) & (ch_labels == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1   = (2 * prec * rec / (prec + rec)
                if (not np.isnan(prec)) and (not np.isnan(rec)) and (prec + rec) > 0
                else float("nan"))
        result[ch] = {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "n_anomalous": int(ch_labels.sum()),
            "n_nominal":   int((ch_labels == 0).sum()),
            "precision": prec, "recall": rec, "f1": f1,
            "fragile": ch in FRAGILE_CHANNELS,
        }
    return result


def main():
    csv_path = os.path.join(_DATA, "segments.csv")
    print("Loading test channel labels...")
    channels, labels = load_test_channels(csv_path)
    labels = list(labels)
    print(f"  Test segments: {len(channels)}")

    all_results = {}

    for display_label, scores_file, result_file, color_key in MODELS:
        scores_path = os.path.join(_RESULTS, scores_file)
        result_path = os.path.join(_RESULTS, result_file)

        if not os.path.exists(scores_path):
            print(f"  SKIP {display_label}: scores file not found ({scores_file})")
            continue
        if not os.path.exists(result_path):
            print(f"  SKIP {display_label}: result JSON not found ({result_file})")
            continue

        print(f"\n[{display_label}]")
        npz   = np.load(scores_path)
        # test_scores is stored under the key "test_scores" or "scores" depending on evaluate.py
        if "test_scores" in npz:
            scores = npz["test_scores"]
        else:
            scores = npz["scores"] if "scores" in npz else None

        if scores is None:
            # Try reconstructing from val + test keys
            print(f"  Available keys: {list(npz.keys())} — trying 'test_scores' fallback")
            continue

        with open(result_path) as f:
            result_json = json.load(f)
        threshold = result_json["metrics"]["threshold"]
        print(f"  Threshold: {threshold:.6f}")
        print(f"  Scores shape: {scores.shape}")

        stats = per_channel_stats(scores, labels, channels, threshold)
        all_results[display_label] = {"color_key": color_key, "stats": stats}

        for ch, s in sorted(stats.items()):
            flag = " [FRAGILE]" if s["fragile"] else ""
            print(f"  {ch}{flag}: TP={s['tp']} FP={s['fp']} TN={s['tn']} FN={s['fn']}"
                  f"  Prec={s['precision']:.3f}  Rec={s['recall']:.3f}  F1={s['f1']:.3f}")

    if not all_results:
        print("\nNo results loaded. Check file paths.")
        return

    # Save JSON
    out_json = os.path.join(_RESULTS, "per_channel_breakdown.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_json}")

    # Save CSV
    rows = []
    for model_label, data in all_results.items():
        for ch, s in data["stats"].items():
            rows.append({
                "model": model_label,
                "channel": ch,
                "fragile": s["fragile"],
                "n_anomalous": s["n_anomalous"],
                "n_nominal": s["n_nominal"],
                "tp": s["tp"], "fp": s["fp"], "tn": s["tn"], "fn": s["fn"],
                "precision": round(s["precision"], 4) if not np.isnan(s["precision"]) else None,
                "recall":    round(s["recall"], 4)    if not np.isnan(s["recall"]) else None,
                "f1":        round(s["f1"], 4)        if not np.isnan(s["f1"]) else None,
            })
    df_out = pd.DataFrame(rows)
    out_csv = os.path.join(_RESULTS, "per_channel_breakdown.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Figure: grouped bar chart of per-channel F1
    channels_sorted = sorted(set(
        ch for data in all_results.values() for ch in data["stats"]
    ))
    n_channels = len(channels_sorted)
    n_models   = len(all_results)
    bar_width  = 0.8 / n_models
    x          = np.arange(n_channels)

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (model_label, data) in enumerate(all_results.items()):
        color_key = data["color_key"]
        color     = WONG.get(color_key, "#888888")
        f1_vals   = []
        for ch in channels_sorted:
            s = data["stats"].get(ch, {})
            f1_vals.append(s.get("f1", float("nan")))
        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, f1_vals, width=bar_width * 0.9,
                      label=model_label, color=color, alpha=0.85)

    # Mark fragile channels
    fragile_idx = [i for i, ch in enumerate(channels_sorted) if ch in FRAGILE_CHANNELS]
    for idx in fragile_idx:
        ax.axvspan(idx - 0.5, idx + 0.5, color="#dddddd", alpha=0.4, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(channels_sorted, rotation=30, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_title("OPS-SAT-AD — Per-Channel F1 at Best-Val-F1 Threshold (Seed 42)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")

    # Footnote about fragile channels
    if fragile_idx:
        ax.text(0.01, 0.01,
                "Shaded channels (CADC0886, CADC0890): single active day each (11 and 14 segments).\n"
                "Per-channel statistics presented for completeness; not reliable estimates of channel-level performance.",
                transform=ax.transAxes, fontsize=6.5, color="#555555",
                verticalalignment="bottom")

    fig.tight_layout()
    for fmt in ("pdf", "png"):
        out_fig = os.path.join(_FIGURES, f"per_channel_f1.{fmt}")
        fig.savefig(out_fig)
        print(f"Saved: {out_fig}")
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
