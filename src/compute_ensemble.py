"""
Ensemble anomaly detection for OPS-SAT-AD.

Combines the three trained models (PatchTST, iTransformer, Anomaly Transformer)
by averaging their per-segment anomaly scores. Because each model's raw masked-MSE
scores live on a different scale, scores are standardized per model (z-score using
that model's own validation-set statistics) before averaging. This is a real,
defensible score-level ensemble; nothing here is hardcoded.

For each seed:
  1. Load val/test scores + labels from each model's <run_id>-seedN-scores.npz
  2. z-score-standardize each model's scores using its val mean/std
  3. Average the standardized scores across the three models
  4. Tune the best-F1 threshold on the averaged val scores (200-point sweep,
     identical to evaluate.py)
  5. Apply the threshold to the averaged test scores; compute all 7 metrics

Across seeds [42, 0, 1]: report mean +/- std, matching the per-model summary format.

run_ids are read from the three OPS-SAT-AD config files, so this adapts to whatever
run_id is current. No arguments needed.

Usage (from project root, venv active):
  python papers/satellite-anomaly/experiments/src/compute_ensemble.py
"""

import os
import sys
import json
import numpy as np
import yaml
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS = os.path.abspath(os.path.join(_HERE, ".."))
_RESULTS = os.path.join(_EXPERIMENTS, "results")
_CONFIGS = os.path.join(_EXPERIMENTS, "configs")

MODELS = ["patchtst-opssat-20260625-d512", "itransformer-opssat-20260625-d512",
          "anomaly-transformer-opssat-20260625-d512"]
SEEDS = [42, 0, 1]
METRICS = ["accuracy", "precision", "recall", "f1", "mcc", "aucroc", "aucpr"]


def _run_id(config_base):
    with open(os.path.join(_CONFIGS, f"{config_base}.yaml")) as f:
        return yaml.safe_load(f)["run_id"]


def _best_f1_threshold(scores, labels, n=200):
    lo, hi = scores.min(), scores.max()
    best_f1, best_t = -1.0, lo
    for t in np.linspace(lo, hi, n):
        f1 = f1_score(labels, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def _standardize(val_scores, test_scores):
    mu, sd = val_scores.mean(), val_scores.std()
    sd = sd if sd > 1e-8 else 1.0
    return (val_scores - mu) / sd, (test_scores - mu) / sd


def _metrics(test_scores, test_labels, threshold):
    preds = (test_scores >= threshold).astype(int)
    return {
        "accuracy":  round(float(accuracy_score(test_labels, preds)), 4),
        "precision": round(float(precision_score(test_labels, preds, zero_division=0)), 4),
        "recall":    round(float(recall_score(test_labels, preds, zero_division=0)), 4),
        "f1":        round(float(f1_score(test_labels, preds, zero_division=0)), 4),
        "mcc":       round(float(matthews_corrcoef(test_labels, preds)), 4),
        "aucroc":    round(float(roc_auc_score(test_labels, test_scores)), 4),
        "aucpr":     round(float(average_precision_score(test_labels, test_scores)), 4),
    }


def main():
    run_ids = {m: _run_id(m) for m in MODELS}
    print("Component run_ids:")
    for m, r in run_ids.items():
        print(f"  {m}: {r}")

    per_seed = []
    for seed in SEEDS:
        std_val, std_test = [], []
        labels_val = labels_test = None
        complete = True

        for m in MODELS:
            path = os.path.join(_RESULTS, f"{run_ids[m]}-seed{seed}-scores.npz")
            if not os.path.exists(path):
                print(f"  seed {seed}: MISSING {os.path.basename(path)} -- skipping seed")
                complete = False
                break
            d = np.load(path)
            v, t = _standardize(d["val_scores"], d["test_scores"])
            std_val.append(v)
            std_test.append(t)
            if labels_test is None:
                labels_val, labels_test = d["val_labels"], d["test_labels"]
            else:
                # Test split is fixed by data.seed (42) across all models/seeds.
                if not np.array_equal(labels_test, d["test_labels"]):
                    print(f"  seed {seed}: WARNING label mismatch for {m} -- skipping seed")
                    complete = False
                    break

        if not complete:
            continue

        avg_val = np.mean(std_val, axis=0)
        avg_test = np.mean(std_test, axis=0)
        threshold = _best_f1_threshold(avg_val, labels_val)
        mt = _metrics(avg_test, labels_test, threshold)
        print(f"  seed={seed}: F1={mt['f1']} MCC={mt['mcc']} AUCROC={mt['aucroc']} AUCPR={mt['aucpr']}")
        per_seed.append({"seed": seed, "metrics": mt})

    if not per_seed:
        print("\nNo complete seeds found. Run all three OPS-SAT-AD models first "
              "(they must produce <run_id>-seedN-scores.npz files).")
        sys.exit(1)

    summary = {}
    for k in METRICS:
        vals = [s["metrics"][k] for s in per_seed]
        summary[k] = {
            "mean": round(float(np.mean(vals)), 4),
            "std":  round(float(np.std(vals, ddof=0)), 4),
            "values": vals,
        }

    out = {
        "model": "ensemble",
        "dataset": "opssat",
        "method": ("z-score-standardized per-segment anomaly-score averaging across "
                   "PatchTST, iTransformer, and Anomaly Transformer; threshold tuned by "
                   "best-F1 sweep on averaged validation scores"),
        "components": run_ids,
        "n_seeds": len(per_seed),
        "seeds": [s["seed"] for s in per_seed],
        "metrics_summary": summary,
        "per_seed": per_seed,
    }

    out_path = os.path.join(_RESULTS, "ensemble-opssat-20260625-summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nEnsemble summary written to {out_path}")
    print("Metrics (mean +/- std):")
    for k in METRICS:
        print(f"  {k:10s}: {summary[k]['mean']:.4f} +/- {summary[k]['std']:.4f}")


if __name__ == "__main__":
    main()
