"""
Step 4f - Anomaly Transformer scoring-variant comparison (no retraining).

Re-scores the NEW d=512 seed-42 AT checkpoint three ways on OPS-SAT-AD and
reports AUCROC + MCC for each:
  1. composite : softmax(-temperature * AssDis) * MSE   (original AT paper score)
  2. assdis    : mean association discrepancy            (masked mean over real timesteps)
  3. mse       : plain masked MSE                         (adopted for OPS-SAT-AD)

For each variant: per-segment scores on val + test, best-F1 threshold from the
200-point val sweep applied to test (identical to evaluate.py), MCC from that
threshold, AUCROC from raw test scores. No point-adjustment.

Usage (from project root, venv active):
  python papers/satellite-anomaly/experiments/src/at_scoring_variants.py
"""

import os
import sys
import json
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS = os.path.abspath(os.path.join(_HERE, ".."))
_RESULTS = os.path.join(_EXPERIMENTS, "results")
_CONFIGS = os.path.join(_EXPERIMENTS, "configs")
_CKPTS = os.path.join(_EXPERIMENTS, "checkpoints")
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data_loader import load_opssat
from models.anomaly_transformer_wrapper import AnomalyTransformerWrapper

CONFIG = os.path.join(_CONFIGS, "anomaly-transformer-opssat-20260625-d512.yaml")
CKPT = os.path.join(_CKPTS, "anomaly-transformer-opssat-20260625-d512-seed42-best.pt")
VARIANTS = ["composite", "assdis", "mse"]


def _best_f1_threshold(scores, labels, n=200):
    lo, hi = float(scores.min()), float(scores.max())
    best_f1, best_t = -1.0, lo
    for t in np.linspace(lo, hi, n):
        f1 = f1_score(labels, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def _rec_and_assdis(wrapper, x):
    """Per-timestep reconstruction MSE and association discrepancy. x: (B, T, 1)."""
    reconstruction, series, prior, _ = wrapper.model(x)
    rec = ((reconstruction - x) ** 2).squeeze(-1)            # (B, T)
    B, T, _ = x.shape
    s_kl = torch.zeros(B, T, device=x.device)
    p_kl = torch.zeros(B, T, device=x.device)
    for u in range(len(prior)):
        prior_norm = prior[u] / prior[u].sum(dim=-1, keepdim=True).clamp(min=1e-8)
        s_kl = s_kl + (series[u] * (torch.log(series[u] + 1e-4)
                                    - torch.log(prior_norm + 1e-4))).sum(-1).mean(1)
        p_kl = p_kl + (prior_norm * (torch.log(prior_norm + 1e-4)
                                     - torch.log(series[u] + 1e-4))).sum(-1).mean(1)
    assdis = s_kl + p_kl                                     # (B, T)
    return rec, assdis


def _score(wrapper, ds, variant, device):
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    scores, labels = [], []
    wrapper.eval()
    with torch.no_grad():
        for segs, masks, lab in loader:
            x = segs.to(device)
            if x.dim() == 2:
                x = x.unsqueeze(-1)
            m = masks.to(device).float()                    # (B, T)
            rec, assdis = _rec_and_assdis(wrapper, x)
            if variant == "mse":
                s = (rec * m).sum(1) / m.sum(1).clamp(min=1)
            elif variant == "assdis":
                s = (assdis * m).sum(1) / m.sum(1).clamp(min=1)
            elif variant == "composite":
                # Original AT paper score: softmax over all T (includes padding),
                # weighting the reconstruction error. This is the form that degrades
                # on heavily-padded OPS-SAT segments.
                metric = torch.softmax(-wrapper.temperature * assdis, dim=-1)
                s = (metric * rec).sum(1)
            else:
                raise ValueError(variant)
            scores.append(s.detach().cpu().numpy())
            labels.append(lab.numpy())
    sc = np.concatenate(scores)
    sc = np.nan_to_num(sc, nan=0.0, posinf=float(np.nanmax(sc[np.isfinite(sc)])) if np.isfinite(sc).any() else 0.0, neginf=0.0)
    return sc, np.concatenate(labels).astype(int)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    with open(CONFIG) as f:
        config = yaml.safe_load(f)

    wrapper = AnomalyTransformerWrapper(config)
    wrapper._wrapper_key = "at"
    ckpt = torch.load(CKPT, map_location=device)
    wrapper.load_state_dict(ckpt["model_state"])
    wrapper = wrapper.to(device).eval()
    print(f"Loaded checkpoint: {os.path.basename(CKPT)} (epoch {ckpt.get('epoch','?')})")

    dc = config["data"]
    csv_path = os.path.join(_EXPERIMENTS, "..", "data", "segments.csv")
    splits = load_opssat(csv_path, T=dc.get("T", 512), min_len=dc.get("min_len", 16),
                         val_frac=dc.get("val_frac", 0.20), seed=dc.get("seed", 42))

    out = {
        "checkpoint": os.path.basename(CKPT),
        "config": os.path.basename(CONFIG),
        "note": ("AT scoring-variant comparison on OPS-SAT-AD, d=512 seed42, no retraining. "
                 "Threshold = best-F1 on val (200 sweep) applied to test; MCC from that threshold; "
                 "AUCROC from raw test scores. No point-adjustment."),
        "variants": {},
    }
    print(f"\n{'variant':12s} {'AUCROC':>8s} {'MCC':>8s} {'threshold':>12s}")
    for v in VARIANTS:
        val_s, val_y = _score(wrapper, splits["val"], v, device)
        test_s, test_y = _score(wrapper, splits["test"], v, device)
        thr = _best_f1_threshold(val_s, val_y)
        preds = (test_s >= thr).astype(int)
        aucroc = float(roc_auc_score(test_y, test_s))
        mcc = float(matthews_corrcoef(test_y, preds))
        out["variants"][v] = {"aucroc": round(aucroc, 4), "mcc": round(mcc, 4),
                              "threshold": float(thr)}
        print(f"{v:12s} {aucroc:8.4f} {mcc:8.4f} {thr:12.6f}")

    out_path = os.path.join(_RESULTS, "at-scoring-variants-20260625.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
