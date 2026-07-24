"""
Stage 6 experiment runner.

Usage (from project root, venv active):
  python papers/satellite-anomaly/experiments/src/run_experiment.py <model> <config> [--seed N]

  model:  anomaly-transformer | patchtst | itransformer
  config: config name without .yaml (e.g. anomaly-transformer-opssat)
  --seed N: override training seed (default: use seed from config)
            When provided, appends -seedN to the run_id so each seed
            produces a separate results file.

Run all three seeds for a model:
  for SEED in 42 0 1; do
    python ... anomaly-transformer anomaly-transformer-opssat --seed $SEED
  done

Then aggregate:
  python papers/satellite-anomaly/experiments/src/summarize_results.py anomaly-transformer-opssat

Outputs (per seed):
  experiments/results/[run-id]-seedN.json
  experiments/logs/[run-id]-seedN.log
  experiments/checkpoints/[run-id]-seedN-best.pt
"""

import sys
import os
import json
import logging
import datetime
import yaml
import numpy as np
import random
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS = os.path.abspath(os.path.join(_HERE, ".."))

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data_loader import load_opssat
from train import train
from evaluate import evaluate


def setup_logger(run_id, logs_dir):
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{run_id}.log")
    logger = logging.getLogger(run_id)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def load_wrapper(model_name, config):
    if model_name == "anomaly-transformer":
        from models.anomaly_transformer_wrapper import AnomalyTransformerWrapper
        return AnomalyTransformerWrapper(config)
    elif model_name == "patchtst":
        from models.patchtst_wrapper import PatchTSTWrapper
        return PatchTSTWrapper(config)
    elif model_name == "itransformer":
        from models.itransformer_wrapper import iTransformerWrapper
        return iTransformerWrapper(config)
    elif model_name == "lstm-ae":
        from models.lstm_ae_wrapper import LSTMAEWrapper
        return LSTMAEWrapper(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    if len(sys.argv) < 3:
        print("Usage: run_experiment.py <model> <config> [--seed N]")
        sys.exit(1)

    model_name = sys.argv[1]
    config_name = sys.argv[2]

    # Parse optional --seed argument
    seed_override = None
    args = sys.argv[3:]
    if "--seed" in args:
        idx = args.index("--seed")
        if idx + 1 >= len(args):
            print("--seed requires a value")
            sys.exit(1)
        seed_override = int(args[idx + 1])

    config_path = os.path.join(_EXPERIMENTS, "configs", f"{config_name}.yaml")
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Apply seed override — affects training seed only; data split is fixed by data.seed in YAML
    if seed_override is not None:
        config["training"]["seed"] = seed_override
        run_id = config["run_id"] + f"-seed{seed_override}"
    else:
        run_id = config["run_id"]
    logs_dir = os.path.join(_EXPERIMENTS, "logs")
    results_dir = os.path.join(_EXPERIMENTS, "results")
    checkpoints_dir = os.path.join(_EXPERIMENTS, "checkpoints")
    os.makedirs(results_dir, exist_ok=True)

    logger = setup_logger(run_id, logs_dir)
    logger.info(f"Run: {run_id}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Config: {config_path}")
    _device_str = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"PyTorch: {torch.__version__} | Device: {_device_str}")

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

    # Seed all RNGs before model initialization — ensures reproducible weight init
    _seed = config["training"]["seed"]
    torch.manual_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)

    # Build wrapper
    logger.info("Building model wrapper...")
    wrapper = load_wrapper(model_name, config)
    n_params = sum(p.numel() for p in wrapper.parameters())
    logger.info(f"Parameters: {n_params:,}")

    # Train
    logger.info("Starting training...")
    best_ckpt, best_epoch, stop_reason = train(
        wrapper=wrapper,
        train_dataset=splits["train"],
        val_dataset=splits["val"],
        config=config,
        run_id=run_id,
        logger=logger,
        checkpoints_dir=checkpoints_dir,
    )

    # Evaluate
    logger.info("Starting evaluation...")
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

    # Write results JSON
    result = {
        "run_id": run_id,
        "model": model_name,
        "dataset": config.get("dataset", "opssat"),
        "seed": config["training"]["seed"],
        "timestamp": datetime.datetime.now().isoformat(),
        "status": "COMPLETE",
        "split_counts": counts,
        "metrics": metrics,
        "best_epoch": best_epoch,
        "stop_reason": stop_reason,
        "config": config,
        "checkpoint": best_ckpt,
        "n_params": n_params,
    }

    result_path = os.path.join(results_dir, f"{run_id}.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Results written to {result_path}")
    logger.info("Run complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
