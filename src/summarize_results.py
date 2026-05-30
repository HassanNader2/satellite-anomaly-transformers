"""
Aggregate per-seed result JSONs into a summary with mean ± std.

Usage (from project root, venv active):
  python papers/satellite-anomaly/experiments/src/summarize_results.py <config-base>

  config-base: config name without .yaml and without -seedN suffix
               e.g. anomaly-transformer-opssat

Looks for results matching: experiments/results/<config-base>-seed*.json
Writes summary to:          experiments/results/<config-base>-summary.json

The summary contains mean and std for all 7 metrics across seeds,
plus the per-seed breakdown for traceability.
"""

import sys
import os
import json
import glob
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = os.path.abspath(os.path.join(_HERE, "..", "results"))

METRICS = ["accuracy", "precision", "recall", "f1", "mcc", "aucroc", "aucpr"]


def main():
    if len(sys.argv) < 2:
        print("Usage: summarize_results.py <config-base>")
        sys.exit(1)

    config_base = sys.argv[1]
    pattern = os.path.join(_RESULTS_DIR, f"{config_base}-seed*.json")
    result_files = sorted(glob.glob(pattern))

    if not result_files:
        print(f"No seed result files found matching: {pattern}")
        sys.exit(1)

    print(f"Found {len(result_files)} seed result(s):")
    per_seed = []
    for path in result_files:
        with open(path) as f:
            data = json.load(f)
        seed = data.get("seed", "?")
        metrics = data["metrics"]
        print(f"  seed={seed}: F1={metrics['f1']} MCC={metrics['mcc']} AUCROC={metrics['aucroc']}")
        per_seed.append({"seed": seed, "run_id": data["run_id"], "metrics": metrics})

    # Compute mean ± std across seeds
    summary_metrics = {}
    for m in METRICS:
        values = [s["metrics"][m] for s in per_seed]
        summary_metrics[m] = {
            "mean": round(float(np.mean(values)), 4),
            "std":  round(float(np.std(values, ddof=0)), 4),
            "values": values,
        }

    summary = {
        "config_base": config_base,
        "n_seeds": len(per_seed),
        "seeds": [s["seed"] for s in per_seed],
        "metrics_summary": summary_metrics,
        "per_seed": per_seed,
    }

    out_path = os.path.join(_RESULTS_DIR, f"{config_base}-summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary written to {out_path}")
    print("\nMetrics (mean ± std):")
    for m in METRICS:
        s = summary_metrics[m]
        print(f"  {m:10s}: {s['mean']:.4f} ± {s['std']:.4f}")


if __name__ == "__main__":
    main()
