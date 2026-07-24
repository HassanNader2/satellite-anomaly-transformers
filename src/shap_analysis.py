"""
Stage 7 — GradientSHAP temporal attribution for all three models.

For each model (AT, PatchTST, iTransformer):
  1. Load seed 42 checkpoint.
  2. Wrap the forward pass as a differentiable scalar scorer (bypasses @no_grad).
  3. Use the 20 shared true-negative segments as SHAP baselines.
  4. Run GradientSHAP on the 20 shared true-positive segments.
  5. Save raw attributions to results/shap_<model>.npz.
  6. Save figures to papers/satellite-anomaly/figures/.

Run from project root with venv active:
  python papers/satellite-anomaly/experiments/src/shap_analysis.py
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from captum.attr import GradientShap

_HERE        = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS = os.path.abspath(os.path.join(_HERE, ".."))
_RESULTS     = os.path.join(_EXPERIMENTS, "results")
_FIGURES     = os.path.abspath(os.path.join(_EXPERIMENTS, "..", "figures"))
os.makedirs(_FIGURES, exist_ok=True)

# Add src to path so model wrappers resolve their own imports
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data_loader import load_opssat

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

WONG = {"at": "#E69F00", "patchtst": "#0072B2", "itransformer": "#009E73"}

MODEL_CFGS = [
    {
        "key":       "at",
        "label":     "Anomaly Transformer",
        "config":    os.path.join(_EXPERIMENTS, "configs", "anomaly-transformer-opssat-20260625-d512.yaml"),
        "ckpt":      os.path.join(_EXPERIMENTS, "checkpoints",
                                  "anomaly-transformer-opssat-20260625-d512-seed42-best.pt"),
        "wrapper":   "anomaly_transformer",
    },
    {
        "key":       "patchtst",
        "label":     "PatchTST",
        "config":    os.path.join(_EXPERIMENTS, "configs", "patchtst-opssat-20260625-d512.yaml"),
        "ckpt":      os.path.join(_EXPERIMENTS, "checkpoints",
                                  "patchtst-opssat-20260625-d512-seed42-best.pt"),
        "wrapper":   "patchtst",
    },
    {
        "key":       "itransformer",
        "label":     "iTransformer",
        "config":    os.path.join(_EXPERIMENTS, "configs", "itransformer-opssat-20260625-d512.yaml"),
        "ckpt":      os.path.join(_EXPERIMENTS, "checkpoints",
                                  "itransformer-opssat-20260625-d512-seed42-best.pt"),
        "wrapper":   "itransformer",
    },
]

N_SHAP_SAMPLES = 50    # GradientSHAP noise samples per segment
STDEVS         = 0.1   # noise std added to baselines


# ---------------------------------------------------------------------------
# Differentiable score wrappers (bypass @no_grad on anomaly_score)
# ---------------------------------------------------------------------------

class _DiffScorer(nn.Module):
    """
    Wraps a model wrapper to produce a differentiable scalar anomaly score.
    mask is fixed per-segment (determined by original length, not values).
    Computes masked mean MSE between reconstruction and input.
    """
    def __init__(self, model_wrapper, mask):
        super().__init__()
        self.mw   = model_wrapper
        # mask: (T,) bool — register as buffer so it moves with the module
        self.register_buffer("mask", mask.float())

    def _score_from_recon(self, x, reconstruction):
        loss = nn.functional.mse_loss(reconstruction, x, reduction="none").squeeze(-1)  # (B, T)
        m = self.mask.unsqueeze(0).expand(x.shape[0], -1)
        return (loss * m).sum(dim=1) / m.sum().clamp(min=1)

    def forward(self, x):
        # x: (B, T, 1) — gradients flow through
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        wrapper_key = getattr(self.mw, "_wrapper_key", None)

        if wrapper_key == "at":
            reconstruction, _, _ = self.mw(x)
        elif wrapper_key in ("patchtst", "itransformer"):
            # Use wrapper.forward() — handles model-specific arg signatures
            # (iTransformer needs x, None, None, None; PatchTST just needs x)
            reconstruction = self.mw(x)
        else:
            # generic fallback
            out = self.mw(x)
            reconstruction = out[0] if isinstance(out, (list, tuple)) else out

        return self._score_from_recon(x, reconstruction)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_wrapper(cfg):
    import yaml
    with open(cfg["config"]) as f:
        config = yaml.safe_load(f)

    key = cfg["wrapper"]
    if key == "anomaly_transformer":
        from models.anomaly_transformer_wrapper import AnomalyTransformerWrapper
        wrapper = AnomalyTransformerWrapper(config)
        wrapper._wrapper_key = "at"
    elif key == "patchtst":
        from models.patchtst_wrapper import PatchTSTWrapper
        wrapper = PatchTSTWrapper(config)
        wrapper._wrapper_key = "patchtst"
    elif key == "itransformer":
        # iTransformer's model dir must be on sys.path before the wrapper import.
        # Flush stale 'model' package cache left by AnomalyTransformer wrapper —
        # both repos use 'model/' as a top-level package and the AT one gets cached first.
        _it_models = os.path.abspath(
            os.path.join(_EXPERIMENTS, "models", "iTransformer")
        )
        if _it_models not in sys.path:
            sys.path.insert(0, _it_models)
        # Both AT and PatchTST use 'model/' and 'layers/' as top-level packages.
        # Flush those caches so iTransformer's versions are loaded cleanly.
        for _k in list(sys.modules.keys()):
            if (_k in ("model", "layers", "utils") or
                    _k.startswith("model.") or
                    _k.startswith("layers.") or
                    _k.startswith("utils.")):
                del sys.modules[_k]
        from models.itransformer_wrapper import iTransformerWrapper
        wrapper = iTransformerWrapper(config)
        wrapper._wrapper_key = "itransformer"
    else:
        raise ValueError(key)

    ckpt = torch.load(cfg["ckpt"], map_location="cpu")
    wrapper.load_state_dict(ckpt["model_state"], strict=False)
    wrapper.eval()
    return wrapper


def _load_test_tensors(data_cfg):
    """Return test_x (N, T, 1) and test_mask (N, T) as float tensors."""
    from torch.utils.data import DataLoader

    csv_path = data_cfg["csv_path"]
    T        = data_cfg.get("T", 512)
    min_len  = data_cfg.get("min_len", 16)
    val_frac = data_cfg.get("val_frac", 0.20)
    seed     = data_cfg.get("seed", 42)

    result   = load_opssat(csv_path, T=T, min_len=min_len,
                           val_frac=val_frac, seed=seed)
    test_ds  = result["test"]

    loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
    x_all, mask_all, _ = next(iter(loader))
    # x_all: (N, T) → add channel dim
    return x_all.float().unsqueeze(-1), mask_all.bool()


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------

def run_shap(scorer, x_tp, mask_tp, x_tn, mask_tn):
    """
    Run GradientSHAP for each TP segment.
    Baselines: all TN segments (with small noise added by captum).

    x_tp:   (N_tp, T, 1)
    x_tn:   (N_tn, T, 1)  — baselines
    mask_tp: (N_tp, T) bool

    Returns attributions: list of (T,) numpy arrays, one per TP segment.
    """
    attributions = []

    for i in range(x_tp.shape[0]):
        seg   = x_tp[i].unsqueeze(0)          # (1, T, 1)
        mask  = mask_tp[i]                      # (T,) bool
        scorer.mask = mask.float().to(seg.device)

        baselines = x_tn.clone()               # (N_tn, T, 1)

        gs = GradientShap(scorer)
        attr = gs.attribute(
            inputs=seg,
            baselines=baselines,
            n_samples=N_SHAP_SAMPLES,
            stdevs=STDEVS,
        )
        # attr: (1, T, 1) → (T,)
        attributions.append(attr.squeeze().detach().numpy())

    return attributions


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _fig_shap_overlay(x_tp, mask_tp, attributions, label, color, key, n_show=4):
    """
    Figure A: n_show segments, each with raw signal + SHAP bar overlay.
    Only non-padded region plotted.
    """
    n_show = min(n_show, len(attributions))
    fig, axes = plt.subplots(n_show, 1, figsize=(7.5, 1.9 * n_show))
    if n_show == 1:
        axes = [axes]

    for ax, i in zip(axes, range(n_show)):
        sig  = x_tp[i].squeeze(-1).numpy()   # (T,)
        attr = attributions[i]                # (T,)
        m    = mask_tp[i].numpy()             # (T,) bool

        t_real = int(m.sum())
        t      = np.arange(t_real)
        sig_r  = sig[:t_real]
        attr_r = attr[:t_real]

        ax2 = ax.twinx()
        ax2.bar(t, attr_r, color=[color if a >= 0 else "#888888" for a in attr_r],
                alpha=0.45, width=1.0, linewidth=0)
        ax2.axhline(0, color="#aaaaaa", linewidth=0.6, linestyle="--")
        ax2.set_ylabel("SHAP", fontsize=7, color="#555555")
        ax2.tick_params(labelsize=6)

        ax.plot(t, sig_r, color="#222222", linewidth=0.9, zorder=3)
        ax.set_xlim(0, t_real)
        ax.set_ylabel("z-score", fontsize=7)
        ax.set_title(f"Sample {i + 1}", fontsize=8, pad=2)

    axes[-1].set_xlabel("Timestep")
    fig.suptitle(f"{label} — SHAP Temporal Attribution (Anomalous Segments)", y=1.01)
    fig.tight_layout()

    for fmt in ("pdf", "png"):
        fig.savefig(os.path.join(_FIGURES, f"shap_{key}_overlay.{fmt}"))
    plt.close(fig)
    print(f"  Saved: shap_{key}_overlay.pdf/.png")


def _fig_mean_attr(attributions_dict, mask_tp):
    """
    Figure B: mean |SHAP| across all TPs for each model, normalised to [0,1].
    All three models on one figure for direct comparison.
    Averaged over the non-padded region only (variable length per segment).
    """
    fig, ax = plt.subplots(figsize=(7.5, 3.0))

    for key, (attributions, label) in attributions_dict.items():
        color = WONG[key]
        # Pad each attribution to T and average, weighted by mask
        T = attributions[0].shape[0]
        stack = np.zeros((len(attributions), T))
        for i, attr in enumerate(attributions):
            stack[i] = np.abs(attr)

        # Mean over segments at each timestep; mask to non-padded region
        m_all = mask_tp.numpy()          # (N, T) bool
        mean_attr = np.where(m_all, stack, np.nan)
        mean_attr = np.nanmean(mean_attr, axis=0)
        # Normalise per model
        mx = mean_attr[~np.isnan(mean_attr)].max()
        if mx > 0:
            mean_attr = mean_attr / mx

        # Compute per-timestep x positions (use median real length as x axis)
        real_len = int(np.median(m_all.sum(axis=1)))
        ax.plot(np.arange(real_len), mean_attr[:real_len],
                color=color, linewidth=1.6, label=label)

    ax.set_xlabel("Timestep (normalised to median segment length)")
    ax.set_ylabel("Normalised mean |SHAP|")
    ax.set_title("OPS-SAT-AD — Mean GradientSHAP Attribution Across 20 Anomalous Segments")
    ax.legend()
    fig.tight_layout()

    for fmt in ("pdf", "png"):
        fig.savefig(os.path.join(_FIGURES, f"shap_mean_attribution.{fmt}"))
    plt.close(fig)
    print("  Saved: shap_mean_attribution.pdf/.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading sample indices...")
    with open(os.path.join(_RESULTS, "shap_sample_indices.json")) as f:
        sample_idx = json.load(f)

    shared_tp = sample_idx["shared_tp"]
    shared_tn = sample_idx["shared_tn"]

    print("Loading test data...")
    import yaml
    with open(MODEL_CFGS[0]["config"]) as f:
        cfg0 = yaml.safe_load(f)
    data_cfg = cfg0["data"]
    # Resolve csv_path relative to experiments dir
    data_cfg["csv_path"] = os.path.join(_EXPERIMENTS, "..", "data", "segments.csv")

    x_test, mask_test = _load_test_tensors(data_cfg)
    print(f"  Test set: {x_test.shape[0]} segments, T={x_test.shape[1]}")

    x_tp   = x_test[shared_tp]        # (20, T, 1)
    mask_tp = mask_test[shared_tp]    # (20, T)
    x_tn   = x_test[shared_tn]        # (20, T, 1)
    mask_tn = mask_test[shared_tn]    # (20, T)

    attributions_dict = {}

    for cfg in MODEL_CFGS:
        print(f"\n[{cfg['label']}]")
        if not os.path.exists(cfg["ckpt"]):
            print(f"  Checkpoint not found: {cfg['ckpt']} — skipping.")
            continue

        print("  Loading model...")
        wrapper = _load_wrapper(cfg)

        # Use mask of first TP segment as placeholder; updated per-segment in run_shap
        scorer = _DiffScorer(wrapper, mask_tp[0])
        scorer.eval()

        print(f"  Running GradientSHAP (n_samples={N_SHAP_SAMPLES})...")
        attributions = run_shap(scorer, x_tp, mask_tp, x_tn, mask_tn)

        # Save raw attributions
        out_npz = os.path.join(_RESULTS, f"shap_{cfg['key']}.npz")
        np.savez(out_npz,
                 attributions=np.array(attributions),
                 shared_tp=np.array(shared_tp),
                 shared_tn=np.array(shared_tn))
        print(f"  Saved: {out_npz}")

        attributions_dict[cfg["key"]] = (attributions, cfg["label"])

        _fig_shap_overlay(x_tp, mask_tp, attributions,
                          label=cfg["label"], color=WONG[cfg["key"]],
                          key=cfg["key"])

    if attributions_dict:
        print("\nGenerating mean attribution comparison figure...")
        _fig_mean_attr(attributions_dict, mask_tp)

    # -----------------------------------------------------------------------
    # Per-segment mean concentration (Table 4 values).
    # For each segment: compute fraction of |attribution| in first 100
    # timesteps (relative to all non-padded positions), and the per-segment
    # maximum absolute attribution. Aggregate: mean +/- std across 20 segments.
    # -----------------------------------------------------------------------
    print("\nComputing per-segment mean concentration (Table 4)...")
    summary = {}
    for key, (attributions, label) in attributions_dict.items():
        concentrations = []
        per_seg_maxima = []
        m_np = mask_tp.numpy()  # (20, T) bool

        for i, attr in enumerate(attributions):
            actual_len = int(m_np[i].sum())
            abs_attr = np.abs(attr)

            total_mass = abs_attr[:actual_len].sum()
            first_100_mass = abs_attr[:min(100, actual_len)].sum()
            conc = float(first_100_mass / total_mass) if total_mass > 1e-12 else 0.0
            concentrations.append(conc)

            seg_max = float(abs_attr[:actual_len].max()) if actual_len > 0 else 0.0
            per_seg_maxima.append(seg_max)

        mean_conc = float(np.mean(concentrations))
        std_conc  = float(np.std(concentrations))
        mean_max  = float(np.mean(per_seg_maxima))
        std_max   = float(np.std(per_seg_maxima))
        ratio     = mean_conc / (1.0 - mean_conc) if mean_conc < 1.0 else float("inf")

        summary[key] = {
            "label":         label,
            "mean_conc":     mean_conc,
            "std_conc":      std_conc,
            "ratio":         ratio,
            "mean_seg_max":  mean_max,
            "std_seg_max":   std_max,
            "concentrations": concentrations,
            "per_seg_maxima": per_seg_maxima,
        }
        print(f"  {label}: conc={mean_conc:.1%} +/- {std_conc:.1%}  ratio={ratio:.2f}x"
              f"  per-seg-max mean={mean_max:.4f} +/- {std_max:.4f}")

    summary_path = os.path.join(_RESULTS, "shap_concentration_20260625.json")
    import json as _json
    with open(summary_path, "w") as f:
        _json.dump(summary, f, indent=2)
    print(f"\n  Saved concentration summary: {summary_path}")
    print("  Use these values to update Table 4 in manuscript.tex.")

    print("\nDone.")


if __name__ == "__main__":
    main()
