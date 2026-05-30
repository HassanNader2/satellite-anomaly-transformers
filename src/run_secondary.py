"""
Stage 8 — Secondary validation on SMAP and MSL.

Runs all three models on SMAP (enc_in=25) and MSL (enc_in=55).

Key differences from OPS-SAT-AD:
  - Continuous multivariate time series, not pre-segmented segments.
  - Sliding window W=100, stride=1 for per-timestep scoring.
  - No padding/masking — all windows are full.
  - AT uses ORIGINAL composite score (softmax(AssDis) × MSE) — restored for
    continuous data where it is designed to operate. Plain MSE was only needed
    for OPS-SAT-AD's heavily-padded pre-segmented format.
  - Metrics: AUCROC and AUCPR (threshold-independent, primary).
    F1 at oracle best threshold (sweep on full test, stated in paper).
    Point-adjustment NOT applied.
  - Threshold: oracle best-F1 on pooled test scores across all channels.

Usage:
  source venv/bin/activate
  cd papers/satellite-anomaly/experiments/src
  python run_secondary.py --dataset smap --model patchtst --seed 42
  python run_secondary.py --dataset msl  --model itransformer --seed 42
  # etc.

All three models × both datasets × seed 42 (secondary validation — single seed acceptable
since this is generalisability demonstration, not the primary benchmark).
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import torch
import yaml
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

_HERE        = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS = os.path.abspath(os.path.join(_HERE, ".."))
_RESULTS     = os.path.join(_EXPERIMENTS, "results")
_LOGS        = os.path.join(_EXPERIMENTS, "logs")

os.makedirs(_RESULTS, exist_ok=True)
os.makedirs(_LOGS,    exist_ok=True)

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data_loader_smap_msl import load_smap_msl, score_continuous

DATASET_CONFIGS = {
    "smap": {
        "base_dir":   os.path.join(_EXPERIMENTS, "..", "data", "smap"),
        "spacecraft": "SMAP",
        "enc_in":     25,
    },
    "msl": {
        "base_dir":   os.path.join(_EXPERIMENTS, "..", "data", "msl"),
        "spacecraft": "MSL",
        "enc_in":     55,
    },
}

MODEL_BASE_CONFIGS = {
    "anomaly-transformer": os.path.join(_EXPERIMENTS, "configs", "anomaly-transformer-opssat.yaml"),
    "patchtst":            os.path.join(_EXPERIMENTS, "configs", "patchtst-opssat.yaml"),
    "itransformer":        os.path.join(_EXPERIMENTS, "configs", "itransformer-opssat.yaml"),
}

WIN_SIZE   = 100
VAL_FRAC   = 0.20
N_THRESH   = 200
BATCH_SIZE = 128


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging(run_id):
    log_path = os.path.join(_LOGS, f"{run_id}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )
    return log_path


def _load_wrapper(model_key, config, dataset_key, seed):
    """Load model wrapper with dataset-specific enc_in and seq_len=WIN_SIZE."""
    enc_in = DATASET_CONFIGS[dataset_key]["enc_in"]

    # Patch config for this dataset
    config = dict(config)
    config["model_params"] = dict(config["model_params"])
    config["model_params"]["enc_in"] = enc_in
    config["model_params"]["seq_len"] = WIN_SIZE

    # PatchTST-specific patches
    if model_key == "patchtst":
        config["model_params"]["pred_len"] = WIN_SIZE
        # Adjust patch_len and stride to fit WIN_SIZE=100
        # patch_len=10, stride=5 → (100-10)/5+1 = 19 patches (clean)
        config["model_params"]["patch_len"] = 10
        config["model_params"]["stride"]    = 5

    # iTransformer
    if model_key == "itransformer":
        config["model_params"]["pred_len"] = WIN_SIZE
        # Larger d_model for real multivariate attention
        config["model_params"]["d_model"]  = 128

    # AT
    if model_key == "anomaly-transformer":
        config["model_params"]["win_size"] = WIN_SIZE
        config["model_params"]["c_out"]    = enc_in

    torch.manual_seed(seed)
    np.random.seed(seed)

    if model_key == "anomaly-transformer":
        from models.anomaly_transformer_wrapper import AnomalyTransformerWrapper
        wrapper = AnomalyTransformerWrapper(config)
        wrapper._wrapper_key = "at"
    elif model_key == "patchtst":
        from models.patchtst_wrapper import PatchTSTWrapper
        wrapper = PatchTSTWrapper(config)
        wrapper._wrapper_key = "patchtst"
    elif model_key == "itransformer":
        _it = os.path.join(_EXPERIMENTS, "models", "iTransformer")
        if _it not in sys.path:
            sys.path.insert(0, _it)
        from models.itransformer_wrapper import iTransformerWrapper
        wrapper = iTransformerWrapper(config)
        wrapper._wrapper_key = "itransformer"

    return wrapper


def _train_channel(wrapper, train_ds, config, model_key):
    """Train on one channel's non-overlapping windows."""
    from torch.utils.data import DataLoader

    training_cfg = config.get("training", {})
    lr       = training_cfg.get("lr", 1e-4)
    epochs   = training_cfg.get("epochs", 10)
    patience = training_cfg.get("patience", 5)

    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    opt    = torch.optim.Adam(wrapper.parameters(), lr=lr)

    best_loss  = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        wrapper.train()
        total_loss = 0.0

        for x in loader:
            # x: (B, W, enc_in)
            opt.zero_grad()

            if model_key == "anomaly-transformer":
                loss1, loss2 = wrapper.compute_train_loss(x)
                loss1.backward(retain_graph=True)
                loss2.backward()
            else:
                loss = wrapper.compute_train_loss(x)
                loss.backward()

            opt.step()

            if model_key == "anomaly-transformer":
                total_loss += loss1.item()
            else:
                total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)

        if avg_loss < best_loss - 1e-6:
            best_loss  = avg_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_loss


def _at_composite_score(wrapper, data: np.ndarray, win_size: int,
                         batch_size: int = 64) -> np.ndarray:
    """
    AT original composite score for continuous data: softmax(-(series_kl+prior_kl)) * MSE.
    Per timestep: mean over all timesteps in the window.
    Restored for SMAP/MSL — the composite score operates correctly on full
    windows without the padding dilution problem of OPS-SAT-AD.
    """
    import torch.nn.functional as F

    wrapper.eval()
    T = len(data)
    n_windows = T - win_size + 1
    all_scores = []

    with torch.no_grad():
        for start in range(0, n_windows, batch_size):
            end   = min(start + batch_size, n_windows)
            batch = np.stack([data[i : i + win_size] for i in range(start, end)])
            x     = torch.from_numpy(batch).float()   # (B, W, enc_in)

            reconstruction, series, prior, _ = wrapper.model(x)
            B, T_w, enc = x.shape

            rec_loss = ((reconstruction - x) ** 2).mean(dim=-1)   # (B, T_w)

            series_kl = torch.zeros(B, T_w)
            prior_kl  = torch.zeros(B, T_w)
            for u in range(len(prior)):
                prior_norm = prior[u] / prior[u].sum(dim=-1, keepdim=True).clamp(min=1e-8)
                s_kl = (series[u] * (torch.log(series[u] + 1e-4)
                                     - torch.log(prior_norm + 1e-4))).sum(dim=-1).mean(dim=1, keepdim=True)
                p_kl = (prior_norm * (torch.log(prior_norm + 1e-4)
                                      - torch.log(series[u] + 1e-4))).sum(dim=-1).mean(dim=1, keepdim=True)
                series_kl += s_kl.squeeze(1)
                prior_kl  += p_kl.squeeze(1)

            metric = F.softmax(-(series_kl + prior_kl) * wrapper.temperature, dim=-1)
            score  = (metric * rec_loss).mean(dim=1)   # (B,)
            all_scores.append(score.numpy())

    scores_per_window = np.concatenate(all_scores)
    scores = np.empty(T)
    scores[:win_size - 1] = scores_per_window[0]
    scores[win_size - 1:] = scores_per_window
    return scores


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(all_scores, all_labels):
    """
    Compute AUCROC, AUCPR, and oracle F1 on concatenated scores/labels.
    Oracle: best F1 across N_THRESH threshold candidates on the same data.
    Stated in paper: 'threshold selected by exhaustive F1 sweep on full test set.'

    Scores are clipped to 99.9th percentile before metric computation to
    prevent extreme outliers (numerical instability on individual channels)
    from dominating the pooled AUCROC ranking. Stated in paper.
    """
    clip_val = min(float(np.percentile(all_scores, 99.9)), 1e6)
    all_scores = np.clip(all_scores, 0, clip_val)

    aucroc = float(roc_auc_score(all_labels, all_scores))
    aucpr  = float(average_precision_score(all_labels, all_scores))

    thresholds = np.linspace(all_scores.min(), all_scores.max(), N_THRESH)
    best_f1, best_thresh = 0.0, thresholds[0]
    for t in thresholds:
        preds = (all_scores >= t).astype(int)
        if preds.sum() == 0:
            continue
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1    = f1
            best_thresh = t

    return {"aucroc": aucroc, "aucpr": aucpr,
            "f1_oracle": float(best_f1), "threshold_oracle": float(best_thresh)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["smap", "msl"])
    parser.add_argument("--model",   required=True,
                        choices=["anomaly-transformer", "patchtst", "itransformer"])
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    run_id = (f"{args.model}-{args.dataset}-{datetime.now().strftime('%Y%m%d')}-01-seed{args.seed}")
    _setup_logging(run_id)
    log = logging.getLogger()

    log.info(f"Run: {run_id}")
    log.info(f"Model: {args.model} | Dataset: {args.dataset} | Seed: {args.seed}")

    # Load base config and adapt
    with open(MODEL_BASE_CONFIGS[args.model]) as f:
        config = yaml.safe_load(f)

    ds_cfg   = DATASET_CONFIGS[args.dataset]
    channels = load_smap_msl(
        base_dir=ds_cfg["base_dir"],
        spacecraft=ds_cfg["spacecraft"],
        win_size=WIN_SIZE,
        val_frac=VAL_FRAC,
    )
    log.info(f"Channels: {len(channels)} | enc_in: {ds_cfg['enc_in']} | win_size: {WIN_SIZE}")

    all_test_scores  = []
    all_test_labels  = []

    for i, ch in enumerate(channels):
        log.info(f"[{i+1}/{len(channels)}] Channel: {ch['chan_id']}")

        wrapper = _load_wrapper(args.model, config, args.dataset, args.seed)

        # Train
        best_loss = _train_channel(wrapper, ch["train_dataset"], config, args.model)
        log.info(f"  Trained — best loss: {best_loss:.6f}")

        # Score test
        wrapper.eval()
        if args.model == "anomaly-transformer":
            test_scores = _at_composite_score(wrapper, ch["test_data"], WIN_SIZE)
        else:
            test_scores = score_continuous(wrapper, ch["test_data"], WIN_SIZE, BATCH_SIZE)

        all_test_scores.append(test_scores)
        all_test_labels.append(ch["test_labels"])

        anom_rate = ch["test_labels"].mean()
        log.info(f"  Test anomaly rate: {anom_rate:.3f} | "
                 f"score range: [{test_scores.min():.4f}, {test_scores.max():.4f}]")

    # Aggregate across all channels
    all_scores = np.concatenate(all_test_scores)
    all_labels = np.concatenate(all_test_labels)

    log.info(f"Pooled test: {len(all_scores)} timesteps | "
             f"anomaly rate: {all_labels.mean():.3f}")

    metrics = _compute_metrics(all_scores, all_labels)
    log.info(f"AUCROC={metrics['aucroc']:.4f} | AUCPR={metrics['aucpr']:.4f} | "
             f"F1(oracle)={metrics['f1_oracle']:.4f}")

    # Save
    result = {
        "run_id":    run_id,
        "model":     args.model,
        "dataset":   args.dataset,
        "seed":      args.seed,
        "timestamp": datetime.now().isoformat(),
        "win_size":  WIN_SIZE,
        "enc_in":    ds_cfg["enc_in"],
        "n_channels": len(channels),
        "n_timesteps": int(len(all_scores)),
        "anomaly_rate": float(all_labels.mean()),
        "metrics":   metrics,
        "note": (
            "AUCROC and AUCPR are primary metrics (threshold-independent). "
            "F1 uses oracle threshold (best-F1 sweep on full test set — stated in paper). "
            "No point-adjustment applied. "
            "AT composite score restored for continuous data (plain MSE used only on OPS-SAT-AD)."
        ),
    }

    out_path = os.path.join(_RESULTS, f"{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"Results saved: {out_path}")

    # Save raw scores
    scores_path = os.path.join(_RESULTS, f"{run_id}-scores.npz")
    np.savez(scores_path, test_scores=all_scores, test_labels=all_labels)
    log.info(f"Scores saved: {scores_path}")


if __name__ == "__main__":
    main()
