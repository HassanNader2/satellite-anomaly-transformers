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

# Secondary validation uses the headline d=512 sweep configs (epochs=150, patience=15,
# e_layers=3, n_heads=8, d_ff=512) so the training rule matches the OPS-SAT runs exactly.
MODEL_BASE_CONFIGS = {
    "anomaly-transformer": os.path.join(_EXPERIMENTS, "configs", "anomaly-transformer-opssat-20260625-d512.yaml"),
    "patchtst":            os.path.join(_EXPERIMENTS, "configs", "patchtst-opssat-20260625-d512.yaml"),
    "itransformer":        os.path.join(_EXPERIMENTS, "configs", "itransformer-opssat-20260625-d512.yaml"),
}

# Sweep tag for the from-scratch 20260625 run (run_ids share this with the OPS-SAT sweep).
DATE_TAG   = "20260625"
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
        # d_model left at the config value so all three models share identical capacity

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


def _train_channel(wrapper, train_ds, val_data, config, model_key, device):
    """Train on one channel under the IDENTICAL rule shared with train.py:
      AdamW + ReduceLROnPlateau(factor=0.5, patience=5, threshold=1e-4 rel, min_lr=1e-6),
      max_epochs=150 (from config), early stop on VALIDATION reconstruction loss with
      relative min_delta=1e-4 at patience=15 (from config), grad clip max_norm=1.0,
      keep the absolute lowest-val-loss state.

    Validation windows are non-overlapping W-length slices of this channel's val_data.
    If the channel has no full validation window, the training loss is used as the
    early-stop signal instead (logged via the returned monitor source).
    """
    from torch.utils.data import DataLoader

    training_cfg = config.get("training", {})
    lr       = training_cfg.get("lr", 1e-4)
    epochs   = training_cfg.get("epochs", 150)
    patience = training_cfg.get("patience", 15)
    wd       = training_cfg.get("weight_decay", 0.0)
    es_rel_min_delta = 1e-4

    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    opt    = torch.optim.AdamW(wrapper.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5,
        threshold=1e-4, threshold_mode="rel", min_lr=1e-6,
    )

    # Build non-overlapping validation windows for the early-stop signal.
    W = WIN_SIZE
    n_val = len(val_data) // W
    val_windows = None
    if n_val >= 1:
        vw = np.stack([val_data[j * W:(j + 1) * W] for j in range(n_val)])
        val_windows = torch.from_numpy(vw).float()

    best_metric = float("inf")
    best_state  = {k: v.detach().cpu().clone() for k, v in wrapper.state_dict().items()}
    no_improve  = 0

    for epoch in range(1, epochs + 1):
        wrapper.train()
        total_loss = 0.0

        for x in loader:
            # x: (B, W, enc_in)
            x = x.float().to(device)
            opt.zero_grad()

            if model_key == "anomaly-transformer":
                # Combined-loss surrogate, matching train.py (single optimizer step).
                loss1, loss2 = wrapper.compute_train_loss(x)
                combined = loss1 + loss2
                combined.backward()
                torch.nn.utils.clip_grad_norm_(wrapper.parameters(), max_norm=1.0)
                opt.step()
                total_loss += loss1.item()
            else:
                loss = wrapper.compute_train_loss(x)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(wrapper.parameters(), max_norm=1.0)
                opt.step()
                total_loss += loss.item()

        avg_train_loss = total_loss / max(len(loader), 1)

        # Validation reconstruction loss (MSE) — early-stop / scheduler signal.
        if val_windows is not None:
            wrapper.eval()
            vloss_sum, nvb = 0.0, 0
            with torch.no_grad():
                for vs in range(0, len(val_windows), BATCH_SIZE):
                    vb = val_windows[vs:vs + BATCH_SIZE].to(device)
                    vloss_sum += wrapper.compute_val_loss(vb).item()
                    nvb += 1
            monitor = vloss_sum / max(nvb, 1)
        else:
            monitor = avg_train_loss

        scheduler.step(monitor)

        # Significant-improvement test against the OLD best (relative min_delta).
        significant = monitor < best_metric * (1.0 - es_rel_min_delta)
        if monitor < best_metric:
            best_metric = monitor
            best_state = {k: v.detach().cpu().clone() for k, v in wrapper.state_dict().items()}
        if significant:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    wrapper.load_state_dict(best_state)
    return best_metric


def _at_composite_score(wrapper, data: np.ndarray, win_size: int,
                         batch_size: int = 64, device=None) -> np.ndarray:
    """
    AT original composite score for continuous data: softmax(-(series_kl+prior_kl)) * MSE.
    Per timestep: mean over all timesteps in the window.
    Restored for SMAP/MSL — the composite score operates correctly on full
    windows without the padding dilution problem of OPS-SAT-AD.
    """
    import torch.nn.functional as F

    if device is None:
        device = next(wrapper.parameters()).device
    wrapper.eval()
    T = len(data)
    n_windows = T - win_size + 1
    all_scores = []

    with torch.no_grad():
        for start in range(0, n_windows, batch_size):
            end   = min(start + batch_size, n_windows)
            batch = np.stack([data[i : i + win_size] for i in range(start, end)])
            x     = torch.from_numpy(batch).float().to(device)   # (B, W, enc_in)

            reconstruction, series, prior, _ = wrapper.model(x)
            B, T_w, enc = x.shape

            rec_loss = ((reconstruction - x) ** 2).mean(dim=-1)   # (B, T_w)

            series_kl = torch.zeros(B, T_w, device=x.device)
            prior_kl  = torch.zeros(B, T_w, device=x.device)
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
            all_scores.append(score.cpu().numpy())

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

    run_id = (f"{args.model}-{args.dataset}-{DATE_TAG}-seed{args.seed}")
    _setup_logging(run_id)
    log = logging.getLogger()

    log.info(f"Run: {run_id}")
    log.info(f"Model: {args.model} | Dataset: {args.dataset} | Seed: {args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

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
        wrapper = wrapper.to(device)

        # Train (early stop on validation reconstruction loss, identical rule)
        best_loss = _train_channel(wrapper, ch["train_dataset"], ch["val_data"], config, args.model, device)
        log.info(f"  Trained — best val loss: {best_loss:.6f}")

        # Score test
        wrapper.eval()
        if args.model == "anomaly-transformer":
            test_scores = _at_composite_score(wrapper, ch["test_data"], WIN_SIZE, device=device)
        else:
            test_scores = score_continuous(wrapper, ch["test_data"], WIN_SIZE, BATCH_SIZE, device=device)

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
