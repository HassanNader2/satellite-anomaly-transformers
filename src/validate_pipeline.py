"""
Stage 5 — Baseline pipeline validation.

Goal: confirm the OPS-SAT-AD data pipeline loads correctly, split counts match
expected values, and one forward pass through Anomaly Transformer completes
without error and produces correct output shapes.

Run from the project root with the venv active:
  python papers/satellite-anomaly/experiments/src/validate_pipeline.py

Writes results to:
  papers/satellite-anomaly/experiments/results/patchtst-opssat-20260514-01.json
  papers/satellite-anomaly/experiments/logs/patchtst-opssat-20260514-01.log
"""

import sys
import os
import json
import logging
import datetime
import torch
from torch.utils.data import DataLoader

# Allow importing from the Anomaly-Transformer repo and from this src/ directory
AT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "Anomaly-Transformer")
SRC_PATH = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(AT_PATH))
sys.path.insert(0, os.path.abspath(SRC_PATH))

from model.AnomalyTransformer import AnomalyTransformer
from data_loader import load_opssat

RUN_ID = "patchtst-opssat-20260514-01"
# __file__ is at experiments/src/validate_pipeline.py
# experiments/ is one level up, project root is four levels up from src/
_HERE = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS = os.path.join(_HERE, "..")
RESULTS_DIR = os.path.join(_EXPERIMENTS, "results")
LOGS_DIR = os.path.join(_EXPERIMENTS, "logs")
CSV_PATH = os.path.join(_HERE, "..", "..", "data", "segments.csv")

T = 512
BATCH_SIZE = 32
SEED = 42

# Expected counts from preprocessing decisions (2026-05-14)
EXPECTED = {
    "filtered_segments": 66,
    "test_total": 509,
    "test_anomalous": 112,
}


def setup_logger(log_path):
    logger = logging.getLogger("validate_pipeline")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    log_path = os.path.join(LOGS_DIR, f"{RUN_ID}.log")
    logger = setup_logger(log_path)

    logger.info(f"Run: {RUN_ID}")
    logger.info(f"Phase: baseline-validate")
    logger.info(f"CSV: {CSV_PATH}")

    # 1. Load data
    logger.info("Loading OPS-SAT-AD data...")
    splits = load_opssat(CSV_PATH, T=T, min_len=16, val_frac=0.20, seed=SEED)
    counts = splits["split_counts"]

    logger.info(f"Filtered segments (< 16 timesteps): {counts['filtered_segments']}")
    logger.info(f"Train: {counts['train_total']} total, {counts['train_anomalous']} anomalous, {counts['train_nominal']} nominal")
    logger.info(f"Val:   {counts['val_total']} total, {counts['val_anomalous']} anomalous, {counts['val_nominal']} nominal")
    logger.info(f"Test:  {counts['test_total']} total, {counts['test_anomalous']} anomalous, {counts['test_nominal']} nominal")

    # 2. Verify counts against expected values
    checks = {}
    for key, expected_val in EXPECTED.items():
        actual = counts[key]
        passed = actual == expected_val
        checks[key] = {"expected": expected_val, "actual": actual, "passed": passed}
        status = "PASS" if passed else "FAIL"
        logger.info(f"Check {key}: expected={expected_val}, actual={actual} [{status}]")

    all_passed = all(c["passed"] for c in checks.values())

    # 3. Build DataLoader and confirm tensor shapes
    train_loader = DataLoader(splits["train"], batch_size=BATCH_SIZE, shuffle=False)
    batch_segs, batch_masks, batch_labels = next(iter(train_loader))

    logger.info(f"Batch shapes — segments: {tuple(batch_segs.shape)}, masks: {tuple(batch_masks.shape)}, labels: {tuple(batch_labels.shape)}")
    assert batch_segs.shape == (BATCH_SIZE, T), f"Unexpected segment shape: {batch_segs.shape}"
    assert batch_masks.shape == (BATCH_SIZE, T), f"Unexpected mask shape: {batch_masks.shape}"
    assert batch_labels.shape == (BATCH_SIZE,), f"Unexpected label shape: {batch_labels.shape}"
    logger.info("Tensor shape check: PASS")

    # 4. Forward pass — Anomaly Transformer (enc_in=1, win_size=T)
    torch.manual_seed(SEED)
    model = AnomalyTransformer(win_size=T, enc_in=1, c_out=1, d_model=64, n_heads=4, e_layers=2, d_ff=128)
    model.eval()

    # Input shape for AT: (batch, seq_len, enc_in) = (B, 512, 1)
    x = batch_segs.unsqueeze(-1)  # (B, T, 1)
    logger.info(f"Model input shape: {tuple(x.shape)}")

    with torch.no_grad():
        output, series, prior, sigmas = model(x)

    logger.info(f"Model output shape: {tuple(output.shape)}")
    assert output.shape == (BATCH_SIZE, T, 1), f"Unexpected output shape: {output.shape}"
    logger.info("Forward pass shape check: PASS")
    logger.info(f"Attention series layers: {len(series)}, prior layers: {len(prior)}")

    # 5. Write results JSON
    result = {
        "run_id": RUN_ID,
        "phase": "baseline-validate",
        "timestamp": datetime.datetime.now().isoformat(),
        "status": "PASS" if all_passed else "FAIL",
        "split_counts": counts,
        "count_checks": checks,
        "shapes": {
            "input": list(x.shape),
            "output": list(output.shape),
        },
        "model": "AnomalyTransformer",
        "params": {
            "T": T, "min_len": 16, "val_frac": 0.20, "seed": SEED,
            "d_model": 64, "n_heads": 4, "e_layers": 2, "d_ff": 128,
        },
    }

    result_path = os.path.join(RESULTS_DIR, f"{RUN_ID}.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Results written to {result_path}")
    logger.info(f"Pipeline validation: {'PASS' if all_passed else 'FAIL'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
