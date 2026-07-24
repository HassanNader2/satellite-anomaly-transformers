"""
Aggregate per-seed SMAP/MSL secondary-validation results into mean +/- std.

run_secondary.py writes per-seed files named:
  <model>-<dataset>-<YYYYMMDD>-01-seed<seed>.json
with metrics: aucroc, aucpr, f1_oracle (NOT the 7 OPS-SAT-AD metrics), so this
is separate from summarize_results.py.

Usage (from project root, venv active):
  python papers/satellite-anomaly/experiments/src/summarize_secondary.py <run-base>

  <run-base>: the run_id without the -seedN suffix,
              e.g. patchtst-smap-20260613-01

Writes: experiments/results/<run-base>-summary.json
"""

import sys
import os
import json
import glob
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = os.path.abspath(os.path.join(_HERE, "..", "results"))

METRICS = ["aucroc", "aucpr", "f1_oracle"]


def main():
    if len(sys.argv) < 2:
        print("Usage: summarize_secondary.py <run-base>  (e.g. patchtst-smap-20260613-01)")
        sys.exit(1)

    run_base = sys.argv[1]
    pattern = os.path.join(_RESULTS_DIR, f"{run_base}-seed*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No seed result files found matching: {pattern}")
        sys.exit(1)

    print(f"Found {len(files)} seed result(s):")
    per_seed = []
    for path in files:
        with open(path) as f:
            data = json.load(f)
        m = data["metrics"]
        seed = data.get("seed", "?")
        print(f"  seed={seed}: AUCROC={m['aucroc']:.4f} AUCPR={m['aucpr']:.4f} F1_oracle={m['f1_oracle']:.4f}")
        per_seed.append({"seed": seed, "run_id": data["run_id"], "metrics": m})

    summary = {}
    for k in METRICS:
        vals = [s["metrics"][k] for s in per_seed]
        summary[k] = {
            "mean": round(float(np.mean(vals)), 4),
            "std":  round(float(np.std(vals, ddof=0)), 4),
            "values": vals,
        }

    out = {
        "run_base": run_base,
        "n_seeds": len(per_seed),
        "seeds": [s["seed"] for s in per_seed],
        "metrics_summary": summary,
        "per_seed": per_seed,
        "note": "Non-point-adjusted. AUCROC/AUCPR are primary; F1 is oracle (best-F1 sweep on full test).",
    }

    out_path = os.path.join(_RESULTS_DIR, f"{run_base}-summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSummary written to {out_path}")
    for k in METRICS:
        print(f"  {k:10s}: {summary[k]['mean']:.4f} +/- {summary[k]['std']:.4f}")


if __name__ == "__main__":
    main()
