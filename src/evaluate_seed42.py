#!/usr/bin/env python
"""
Evaluate seed 42 checkpoint directly (training was incomplete).
Loads the best checkpoint from epoch 3 and evaluates on val/test sets.
"""

import sys
import os
import json
import logging
import yaml
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS = os.path.abspath(os.path.join(_HERE, ".."))

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data_loader import load_opssat
from evaluate import evaluate
from models.anomaly_transformer_wrapper import AnomalyTransformerWrapper


def main():
    config_path = os.path.join(_EXPERIMENTS, "configs", "anomaly-transformer-opssat.yaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"[DEBUG] Config keys: {list(config.keys())}")
    print(f"[DEBUG] model_params in config: {'model_params' in config}")

    run_id = "anomaly-transformer-opssat-20260516-d64-01-seed42"

    # Setup logger
    logs_dir = os.path.join(_EXPERIMENTS, "logs")
    log_path = os.path.join(logs_dir, f"{run_id}-eval.log")
    logger = logging.getLogger(run_id)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)
        logger.addHandler(sh)

    logger.info(f"=== Evaluating {run_id} ===")

    # Load data
    data_cfg = config["data"]
    csv_path = os.path.join(_EXPERIMENTS, "..", "data", "segments.csv")
    logger.info(f"Loading data from {csv_path}")
    splits = load_opssat(
        csv_path,
        T=data_cfg.get("T", 512),
        min_len=data_cfg.get("min_len", 16),
        val_frac=data_cfg.get("val_frac", 0.20),
        seed=data_cfg.get("seed", 42),
    )
    counts = splits["split_counts"]
    logger.info(f"Train: {counts['train_total']} | Val: {counts['val_total']} | Test: {counts['test_total']}")

    # Load wrapper
    logger.info("Building model wrapper...")
    wrapper = AnomalyTransformerWrapper(config)
    n_params = sum(p.numel() for p in wrapper.parameters())
    logger.info(f"Parameters: {n_params:,}")

    # Evaluate with best checkpoint from epoch 3
    logger.info("Starting evaluation (using epoch 3 checkpoint)...")
    checkpoints_dir = os.path.join(_EXPERIMENTS, "checkpoints")
    best_ckpt = os.path.join(checkpoints_dir, f"{run_id}-best.pt")
    results_dir = os.path.join(_EXPERIMENTS, "results")
    scores_path = os.path.join(results_dir, f"{run_id}-scores.npz")

    metrics = evaluate(
        wrapper=wrapper,
        checkpoint_path=best_ckpt,
        val_dataset=splits["val"],
        test_dataset=splits["test"],
        config=config,
        logger=logger,
        scores_path=scores_path,
    )

    logger.info("\n=== RESULTS ===")
    for key in ["accuracy", "precision", "recall", "f1", "mcc", "aucroc", "aucpr"]:
        logger.info(f"{key}: {metrics[key]:.4f}")

    # Write results JSON
    result = {
        "run_id": run_id,
        "model": "anomaly-transformer",
        "dataset": config.get("dataset", "opssat"),
        "seed": 42,
        "status": "COMPLETE (using epoch 3 checkpoint)",
        "split_counts": counts,
        "metrics": metrics,
        "config": config,
        "checkpoint": best_ckpt,
        "n_params": n_params,
    }

    result_path = os.path.join(results_dir, f"{run_id}.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"\nResults written to {result_path}")
    logger.info("Evaluation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
