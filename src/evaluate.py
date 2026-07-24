"""
Evaluation: threshold tuning on val set, then all 7 metrics on test set.

Threshold selection: best-F1 sweep over 200 evenly-spaced values between
min and max of val anomaly scores. Threshold applied identically to test set.

All 7 OPS-SAT-AD metrics (segment-level classification — PA does not apply):
  Accuracy, Precision, Recall, F1, MCC, AUCROC, AUCPR
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
)


def _collect_scores(wrapper, dataset, batch_size=64, device=torch.device("cpu")):
    """Score every segment in a dataset. Returns (scores_np, labels_np)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    all_scores = []
    all_labels = []

    wrapper.eval()
    with torch.no_grad():
        for segs, masks, labels in loader:
            segs = segs.to(device)
            masks = masks.to(device)
            scores = wrapper.anomaly_score(segs, masks)
            all_scores.append(scores.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)


def _best_f1_threshold(scores, labels, n_thresholds=200):
    """Sweep thresholds; return the one that maximises F1 on the provided set."""
    lo, hi = scores.min(), scores.max()
    thresholds = np.linspace(lo, hi, n_thresholds)
    best_f1 = -1.0
    best_thresh = lo

    for t in thresholds:
        preds = (scores >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return best_thresh, best_f1


def evaluate(wrapper, checkpoint_path, val_dataset, test_dataset, config,
             logger=None, scores_path=None):
    """
    Load checkpoint, tune threshold on val set, compute all 7 metrics on test set.

    If scores_path is provided, raw test/val scores and labels are saved as .npz
    alongside the results JSON — required for ROC/PR curve generation.

    Returns a metrics dict ready to be written to the results JSON.
    """
    import logging
    if logger is None:
        logger = logging.getLogger("evaluate")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluation device: {device}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    wrapper.load_state_dict(ckpt["model_state"])
    wrapper = wrapper.to(device)
    logger.info(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.6f})")

    batch_size = config["training"].get("batch_size", 64)

    # Score val set — threshold tuning
    logger.info("Scoring validation set...")
    val_scores, val_labels = _collect_scores(wrapper, val_dataset, batch_size, device)
    threshold, val_f1 = _best_f1_threshold(val_scores, val_labels)
    logger.info(f"Best val threshold={threshold:.6f} (val_F1={val_f1:.4f})")

    # Score test set
    logger.info("Scoring test set...")
    test_scores, test_labels = _collect_scores(wrapper, test_dataset, batch_size, device)
    test_preds = (test_scores >= threshold).astype(int)

    # All 7 metrics
    acc   = accuracy_score(test_labels, test_preds)
    prec  = precision_score(test_labels, test_preds, zero_division=0)
    rec   = recall_score(test_labels, test_preds, zero_division=0)
    f1    = f1_score(test_labels, test_preds, zero_division=0)
    mcc   = matthews_corrcoef(test_labels, test_preds)
    aucroc = roc_auc_score(test_labels, test_scores)
    aucpr  = average_precision_score(test_labels, test_scores)

    metrics = {
        "accuracy":  round(float(acc),    4),
        "precision": round(float(prec),   4),
        "recall":    round(float(rec),    4),
        "f1":        round(float(f1),     4),
        "mcc":       round(float(mcc),    4),
        "aucroc":    round(float(aucroc), 4),
        "aucpr":     round(float(aucpr),  4),
        "threshold": round(float(threshold), 6),
        "val_f1":    round(float(val_f1), 4),
    }

    logger.info("Test metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    if scores_path is not None:
        np.savez(
            scores_path,
            test_scores=test_scores,
            test_labels=test_labels,
            val_scores=val_scores,
            val_labels=val_labels,
            threshold=np.array([threshold]),
        )
        logger.info(f"Raw scores saved to {scores_path}")

    return metrics
