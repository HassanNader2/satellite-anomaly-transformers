"""
Inference latency benchmark for the three OPS-SAT-AD models.

Loads each model's seed-42 checkpoint, runs warmup passes, then times real
forward passes through anomaly_score(). Reports per-segment latency (batch size 1)
and batched throughput. Runs on GPU if available, else CPU. Nothing is hardcoded --
all numbers are measured.

The three cloned model repos share top-level package names (layers, model), so
importing more than one in a single process collides. To avoid that, each model is
benchmarked in its own subprocess (this script re-invokes itself with a model name).

run_id and model_params are read from the config files. No arguments needed for the
normal (all-models) run.

Usage (from project root, venv active):
  python papers/satellite-anomaly/experiments/src/benchmark_inference.py
"""

import os
import sys
import json
import time
import subprocess
import numpy as np
import yaml
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_EXPERIMENTS = os.path.abspath(os.path.join(_HERE, ".."))
_RESULTS = os.path.join(_EXPERIMENTS, "results")
_CONFIGS = os.path.join(_EXPERIMENTS, "configs")
_CKPTS = os.path.join(_EXPERIMENTS, "checkpoints")

MODELS = {
    "patchtst": "patchtst-opssat-20260625-d512",
    "itransformer": "itransformer-opssat-20260625-d512",
    "anomaly-transformer": "anomaly-transformer-opssat-20260625-d512",
}
N_WARMUP = 20
N_ITERS = 200
BATCH_THROUGHPUT = 64

_MARKER = "RESULT_JSON:"


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time(wrapper, x, mask, device, n):
    times = []
    with torch.no_grad():
        for _ in range(n):
            _sync(device)
            t0 = time.perf_counter()
            wrapper.anomaly_score(x, mask)
            _sync(device)
            times.append(time.perf_counter() - t0)
    return np.array(times)


def benchmark_one(model_name):
    """Benchmark a single model in this (isolated) process. Returns dict or None."""
    from run_experiment import load_wrapper  # local import: only this model's repo loads

    config_base = MODELS[model_name]
    with open(os.path.join(_CONFIGS, f"{config_base}.yaml")) as f:
        config = yaml.safe_load(f)
    mp = config["model_params"]
    T = mp.get("seq_len", mp.get("win_size"))
    run_id = config["run_id"]
    ckpt_path = os.path.join(_CKPTS, f"{run_id}-seed42-best.pt")

    if not os.path.exists(ckpt_path):
        print(f"  {model_name}: MISSING checkpoint {os.path.basename(ckpt_path)} -- skipping")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = load_wrapper(model_name, config)
    ckpt = torch.load(ckpt_path, map_location=device)
    wrapper.load_state_dict(ckpt["model_state"])
    wrapper = wrapper.to(device).eval()

    x1 = torch.randn(1, T, 1, device=device)
    m1 = torch.ones(1, T, device=device)
    _time(wrapper, x1, m1, device, N_WARMUP)
    lat = _time(wrapper, x1, m1, device, N_ITERS) * 1000.0  # ms

    xb = torch.randn(BATCH_THROUGHPUT, T, 1, device=device)
    mb = torch.ones(BATCH_THROUGHPUT, T, device=device)
    _time(wrapper, xb, mb, device, max(N_WARMUP // 2, 1))
    bt = _time(wrapper, xb, mb, device, max(N_ITERS // 2, 1))
    seg_per_s = BATCH_THROUGHPUT / bt.mean()

    res = {
        "latency_ms_per_segment_mean": round(float(lat.mean()), 4),
        "latency_ms_per_segment_std": round(float(lat.std()), 4),
        "throughput_segments_per_sec": round(float(seg_per_s), 1),
        "throughput_batch_size": BATCH_THROUGHPUT,
        "seq_len": int(T),
    }
    print(f"  {model_name}: {lat.mean():.3f} +/- {lat.std():.3f} ms/segment "
          f"(batch=1) | {seg_per_s:.0f} seg/s (batch={BATCH_THROUGHPUT})")
    return res


def main_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Device: {device} ({gpu_name})")

    results = {
        "device": str(device),
        "gpu_name": gpu_name,
        "n_warmup": N_WARMUP,
        "n_iters": N_ITERS,
        "models": {},
    }

    for model_name in MODELS:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), model_name],
            capture_output=True, text=True,
        )
        for line in proc.stdout.splitlines():
            if line.startswith(_MARKER):
                payload = line[len(_MARKER):]
                if payload != "null":
                    results["models"][model_name] = json.loads(payload)
            else:
                print(line)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)

    if not results["models"]:
        print("\nNo checkpoints found. Run all three OPS-SAT-AD models (seed 42) first.")
        sys.exit(1)

    out_path = os.path.join(_RESULTS, "inference-benchmark-20260625.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nInference benchmark written to {out_path}")


def main_single(model_name):
    if model_name not in MODELS:
        print(f"Unknown model: {model_name}")
        sys.exit(2)
    res = benchmark_one(model_name)
    print(_MARKER + (json.dumps(res) if res is not None else "null"))


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        main_single(sys.argv[1])
    else:
        main_all()
