"""
Stage 7 — PatchTST patch-level attention weight visualization.

PatchTST has native store_attn support. After a forward pass with store_attn=True,
attention weights are stored at model.backbone.encoder.encoder.layers[i].attn
with shape (B, n_heads, n_patches, n_patches).

For OPS-SAT-AD: seq_len=512, patch_len=16, stride=8, padding_patch='end' → n_patches=64.

Generates:
  1. attention_heatmap_tp.pdf/.png — mean attention across heads, layer 1 and 2,
     averaged over the 20 shared TPs. Shows which patches attend to which patches.
  2. attention_tp_vs_tn.pdf/.png — mean attention weight received per patch
     (column-sum) for anomalous vs. nominal segments. Shows where the model
     focuses for anomalies vs. normal patterns.

Run from project root with venv active:
  python papers/satellite-anomaly/experiments/src/attention_viz.py
"""

import os
import sys
import json
import importlib.util
import types
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE        = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS = os.path.abspath(os.path.join(_HERE, ".."))
_RESULTS     = os.path.join(_EXPERIMENTS, "results")
_FIGURES     = os.path.abspath(os.path.join(_EXPERIMENTS, "..", "figures"))
os.makedirs(_FIGURES, exist_ok=True)

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data_loader import load_opssat, OPSSATDataset

_PATCHTST_SUPERVISED = os.path.join(
    _EXPERIMENTS, "models", "PatchTST", "PatchTST_supervised"
)
if _PATCHTST_SUPERVISED not in sys.path:
    sys.path.insert(0, _PATCHTST_SUPERVISED)

matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

CKPT   = os.path.join(_EXPERIMENTS, "checkpoints",
                      "patchtst-opssat-20260602-01-seed42-best.pt")
CONFIG = os.path.join(_EXPERIMENTS, "configs", "patchtst-opssat.yaml")


# ---------------------------------------------------------------------------
# Build PatchTST with store_attn=True
# ---------------------------------------------------------------------------

def _build_patchtst_with_attn(config: dict):
    spec = importlib.util.spec_from_file_location(
        "patchtst_model",
        os.path.join(_PATCHTST_SUPERVISED, "models", "PatchTST.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    Model = mod.Model

    p = config["model_params"]
    cfg = types.SimpleNamespace(
        enc_in=p["enc_in"],
        seq_len=p["seq_len"],
        pred_len=p["seq_len"],
        e_layers=p.get("e_layers", 2),
        n_heads=p.get("n_heads", 4),
        d_model=p.get("d_model", 128),
        d_ff=p.get("d_ff", 256),
        dropout=p.get("dropout", 0.1),
        fc_dropout=p.get("fc_dropout", 0.1),
        head_dropout=p.get("head_dropout", 0.0),
        individual=p.get("individual", False),
        patch_len=p.get("patch_len", 16),
        stride=p.get("stride", 8),
        padding_patch=p.get("padding_patch", "end"),
        revin=False,
        affine=p.get("affine", False),
        subtract_last=p.get("subtract_last", False),
        decomposition=p.get("decomposition", False),
        kernel_size=p.get("kernel_size", 25),
    )
    model = Model(cfg, store_attn=True)
    return model


def _get_attention(model):
    """
    Extract stored attention weights from all encoder layers.
    Returns list of (n_heads, n_patches, n_patches) arrays, one per layer.
    """
    attn_list = []
    # Access path: model.backbone.encoder.encoder.layers[i].attn
    try:
        layers = model.backbone.encoder.encoder.layers
    except AttributeError:
        # Fallback: walk the module tree looking for TSTEncoderLayer instances
        layers = [m for m in model.modules()
                  if m.__class__.__name__ == "TSTEncoderLayer"]

    for layer in layers:
        if hasattr(layer, "attn") and layer.attn is not None:
            # attn: (B, n_heads, n_patches, n_patches) — B=1 here
            attn_list.append(layer.attn.squeeze(0).detach().numpy())
    return attn_list


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import yaml
    print("Loading config and checkpoint...")
    with open(CONFIG) as f:
        config = yaml.safe_load(f)

    model = _build_patchtst_with_attn(config)
    ckpt  = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    print("  Model loaded.")

    print("Loading sample indices...")
    with open(os.path.join(_RESULTS, "shap_sample_indices.json")) as f:
        sample_idx = json.load(f)
    shared_tp = sample_idx["shared_tp"]
    shared_tn = sample_idx["shared_tn"]

    print("Loading test data...")
    data_cfg = config["data"]
    data_cfg["csv_path"] = os.path.join(_EXPERIMENTS, "..", "data", "segments.csv")
    from torch.utils.data import DataLoader
    result = load_opssat(
        data_cfg["csv_path"],
        T=data_cfg.get("T", 512),
        min_len=data_cfg.get("min_len", 16),
        val_frac=data_cfg.get("val_frac", 0.20),
        seed=data_cfg.get("seed", 42),
    )
    test_ds = result["test"]
    loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
    x_all, _, _ = next(iter(loader))
    # x_all: (N, T) → add channel dim → (N, T, 1)
    x_all = x_all.float().unsqueeze(-1)

    x_tp = x_all[shared_tp]   # (20, T, 1)
    x_tn = x_all[shared_tn]   # (20, T, 1)

    # Compute n_patches
    seq_len   = config["model_params"]["seq_len"]
    patch_len = config["model_params"].get("patch_len", 16)
    stride    = config["model_params"].get("stride", 8)
    # n_patches derived from actual attention shape after forward pass (accounts for padding_patch='end')
    n_layers  = config["model_params"].get("e_layers", 2)
    n_heads   = config["model_params"].get("n_heads", 4)
    n_patches = None  # set after first forward pass

    # --- Collect attention weights ---
    print("Extracting attention weights for TP segments...")
    attn_tp = []   # list of lists: [layer][segment] = (n_heads, n_patches, n_patches)
    for i in range(x_tp.shape[0]):
        with torch.no_grad():
            model(x_tp[i].unsqueeze(0))
        layers_attn = _get_attention(model)
        attn_tp.append(layers_attn)

    print("Extracting attention weights for TN segments...")
    attn_tn = []
    for i in range(x_tn.shape[0]):
        with torch.no_grad():
            model(x_tn[i].unsqueeze(0))
        layers_attn = _get_attention(model)
        attn_tn.append(layers_attn)

    if not attn_tp or not attn_tp[0]:
        print("ERROR: No attention weights captured. Check store_attn path.")
        return

    n_layers_found = len(attn_tp[0])
    # Derive actual n_patches from attention tensor shape
    n_patches = attn_tp[0][0].shape[-1]
    print(f"  Found {n_layers_found} layer(s) with stored attention.")
    print(f"  n_patches (from attention shape)={n_patches}, e_layers={n_layers}, n_heads={n_heads}")

    # --- Figure 1: Mean attention heatmaps (TP) ---
    print("Generating attention heatmap figure...")
    fig, axes = plt.subplots(1, n_layers_found, figsize=(4.5 * n_layers_found, 4.0))
    if n_layers_found == 1:
        axes = [axes]

    for layer_idx in range(n_layers_found):
        # Mean over segments and heads: (n_patches, n_patches)
        mean_attn = np.mean(
            [seg[layer_idx].mean(axis=0) for seg in attn_tp], axis=0
        )
        ax = axes[layer_idx]
        im = ax.imshow(mean_attn, aspect="auto", cmap="Blues",
                       vmin=0, vmax=mean_attn.max())
        ax.set_title(f"Layer {layer_idx + 1}")
        ax.set_xlabel("Key patch index")
        if layer_idx == 0:
            ax.set_ylabel("Query patch index")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"PatchTST — Mean Attention Weights (Anomalous Segments, n={len(shared_tp)})\n"
        f"patch_len={patch_len}, stride={stride}, n_patches={n_patches}",
        y=1.01
    )
    fig.tight_layout()
    for fmt in ("pdf", "png"):
        fig.savefig(os.path.join(_FIGURES, f"attention_heatmap_tp.{fmt}"))
    plt.close(fig)
    print("  Saved: attention_heatmap_tp.pdf/.png")

    # --- Figure 2: Received attention per patch, TP vs TN ---
    print("Generating TP vs TN attention comparison figure...")
    fig, axes = plt.subplots(1, n_layers_found, figsize=(4.5 * n_layers_found, 3.5))
    if n_layers_found == 1:
        axes = [axes]

    for layer_idx in range(n_layers_found):
        # Column-sum of attention = attention RECEIVED by each patch
        recv_tp = np.mean(
            [seg[layer_idx].mean(axis=0).sum(axis=0) for seg in attn_tp], axis=0
        )
        recv_tn = np.mean(
            [seg[layer_idx].mean(axis=0).sum(axis=0) for seg in attn_tn], axis=0
        )
        # Normalise
        recv_tp = recv_tp / recv_tp.sum()
        recv_tn = recv_tn / recv_tn.sum()

        ax = axes[layer_idx]
        patch_idx = np.arange(n_patches)
        # Timestep corresponding to patch centre
        patch_t   = patch_idx * stride + patch_len // 2

        ax.plot(patch_t, recv_tp, color="#0072B2", linewidth=1.6, label="Anomalous")
        ax.plot(patch_t, recv_tn, color="#888888", linewidth=1.2,
                linestyle="--", alpha=0.8, label="Nominal")
        ax.set_title(f"Layer {layer_idx + 1}")
        ax.set_xlabel("Patch centre (timestep)")
        if layer_idx == 0:
            ax.set_ylabel("Normalised attention received")
        ax.legend()

    fig.suptitle(
        "PatchTST — Attention Received per Patch: Anomalous vs Nominal",
        y=1.01
    )
    fig.tight_layout()
    for fmt in ("pdf", "png"):
        fig.savefig(os.path.join(_FIGURES, f"attention_tp_vs_tn.{fmt}"))
    plt.close(fig)
    print("  Saved: attention_tp_vs_tn.pdf/.png")

    # --- Entropy computation ---
    # H = -sum(p * log(p)) over column-sum-normalised attention weights.
    # Computed per-layer for TP and TN. Maximum entropy for 64 patches = ln(64) = 4.159.
    eps = 1e-12
    entropy_results = {}
    print("Computing attention entropy...")
    for layer_idx in range(n_layers_found):
        recv_tp_all = np.array(
            [seg[layer_idx].mean(axis=0).sum(axis=0) for seg in attn_tp]
        )  # (n_samples, n_patches)
        recv_tn_all = np.array(
            [seg[layer_idx].mean(axis=0).sum(axis=0) for seg in attn_tn]
        )
        # Normalise each sample independently
        recv_tp_norm = recv_tp_all / (recv_tp_all.sum(axis=1, keepdims=True) + eps)
        recv_tn_norm = recv_tn_all / (recv_tn_all.sum(axis=1, keepdims=True) + eps)
        # Per-sample entropy then mean
        h_tp = -(recv_tp_norm * np.log(recv_tp_norm + eps)).sum(axis=1)
        h_tn = -(recv_tn_norm * np.log(recv_tn_norm + eps)).sum(axis=1)
        key = f"layer{layer_idx + 1}"
        entropy_results[key] = {
            "tp_mean": float(h_tp.mean()),
            "tp_std":  float(h_tp.std()),
            "tn_mean": float(h_tn.mean()),
            "tn_std":  float(h_tn.std()),
            "max_entropy": float(np.log(n_patches)),
        }
        print(f"  Layer {layer_idx + 1}: TP entropy={h_tp.mean():.4f}±{h_tp.std():.4f}  "
              f"TN entropy={h_tn.mean():.4f}±{h_tn.std():.4f}  "
              f"(max={np.log(n_patches):.4f})")

    import json as _json
    entropy_json = os.path.join(_RESULTS, "attention_entropy.json")
    with open(entropy_json, "w") as fj:
        _json.dump({"n_patches": n_patches, "n_tp": len(attn_tp),
                    "n_tn": len(attn_tn), "layers": entropy_results}, fj, indent=2)
    print(f"  Entropy saved: {entropy_json}")

    # Save raw attention arrays
    out_npz = os.path.join(_RESULTS, "attention_patchtst.npz")
    np.savez(
        out_npz,
        attn_tp_l1=np.array([s[0] for s in attn_tp]) if n_layers_found >= 1 else np.array([]),
        attn_tp_l2=np.array([s[1] for s in attn_tp]) if n_layers_found >= 2 else np.array([]),
        attn_tn_l1=np.array([s[0] for s in attn_tn]) if n_layers_found >= 1 else np.array([]),
        attn_tn_l2=np.array([s[1] for s in attn_tn]) if n_layers_found >= 2 else np.array([]),
        shared_tp=np.array(shared_tp),
        shared_tn=np.array(shared_tn),
    )
    print(f"  Saved: {out_npz}")
    print("\nDone.")


if __name__ == "__main__":
    main()
