"""
Incrementally append the from-scratch (20260625) run manifest to GPU_SESSION_LOG.md.

Usage:
  python append_manifest.py --header           # write the manifest header once
  python append_manifest.py --group <cfg-base> # append a 3-seed group section once
  python append_manifest.py --secondary <cfg-base>  # append a secondary 3-seed group

Idempotent: each section is keyed by a unique marker and skipped if already present.
All writes are UTF-8.
"""

import os
import sys
import json
import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.abspath(os.path.join(_HERE, ".."))
_RESULTS = os.path.join(_EXP, "results")
_LOG = os.path.abspath(os.path.join(_EXP, "..", "GPU_SESSION_LOG.md"))

HEADER_MARKER = "## From-scratch run manifest (20260625)"

HEADER = """## From-scratch run manifest (20260625)

**Started:** {ts}
**GPU:** NVIDIA GeForce RTX 4060 Laptop (8 GB) | PyTorch 2.5.1+cu121 | device=cuda

This section is appended incrementally: one block per three-seed group as it finishes.

### Step 0 — identical training rule (all runs, OPS-SAT and secondary)

- Optimizer: AdamW (weight_decay from config, default 0.0), gradient clipping max_norm=1.0.
- LR schedule: ReduceLROnPlateau(mode=min, factor=0.5, patience=5, threshold=1e-4,
  threshold_mode='rel', min_lr=1e-6), stepped on validation reconstruction loss each epoch.
- Early stopping: max_epochs=150, patience=15 on validation reconstruction loss with
  relative min_delta=1e-4 (an epoch resets patience only if val loss drops by >0.01% of the
  running best; otherwise it counts toward patience). The absolute lowest-val-loss checkpoint
  is always kept.
- "Optimized epochs" = early stopping on validation loss, never test AUCROC (no leakage).
- Threshold: val-set best-F1 sweep over 200 thresholds, applied unchanged to test.
- Training is unsupervised (no labels in loss or early-stop signal).
- Code: train.py (OPS-SAT); run_secondary.py inner per-channel loop (SMAP/MSL, same rule).

### Sweep design

- OPS-SAT-AD 4x4: models {{patchtst, itransformer, anomaly-transformer (d_model);
  lstm-ae (hidden_size)}} x capacities {{64,128,256,512}} x seeds {{42,0,1}} = 48 runs.
- Held constant: n_heads=8, e_layers=3, d_ff=d_model, dropout=0.1, batch_size=16, lr=1e-4,
  max_epochs=150, patience=15. Configs: experiments/configs/<model>-opssat-20260625-d<CAP>.yaml.
- Run IDs: <model>-opssat-20260625-d<CAP>-seed<N>. Summaries: ...-d<CAP>-summary.json.
- Secondary (after OPS-SAT): PatchTST/iTransformer/AnomalyTransformer x {{SMAP,MSL}} x
  seeds {{42,0,1}} = 18 runs at d=512, per-channel, no point-adjustment.
- Step 1: prior results (20260613 and earlier) archived to results/archive_pre20260625/.

### Completed three-seed groups
"""

GROUP_METRICS = [("f1", "F1"), ("mcc", "MCC"), ("aucroc", "AUCROC"),
                 ("aucpr", "AUCPR"), ("accuracy", "Accuracy")]
SECONDARY_METRICS = [("aucroc", "AUCROC"), ("aucpr", "AUCPR"), ("f1_oracle", "F1(oracle)")]

OPSSAT_MODELS = ["itransformer", "patchtst", "lstm-ae", "anomaly-transformer"]
OPSSAT_CAPS = [64, 128, 256, 512]
OPSSAT_FINAL_MARKER = "### OPS-SAT-AD 4x4 capacity sweep — FINAL (20260625)"


def _read():
    if not os.path.exists(_LOG):
        return ""
    with open(_LOG, "r", encoding="utf-8") as f:
        return f.read()


def _append(text):
    with open(_LOG, "a", encoding="utf-8") as f:
        f.write(text)


def write_header():
    if HEADER_MARKER in _read():
        print("Header already present; skipping.")
        return
    _append("\n\n---\n\n" + HEADER.format(ts=datetime.datetime.now().isoformat(timespec="seconds")))
    print("Header appended to GPU_SESSION_LOG.md")


def _group_block(cfg, summ, kind):
    ms = summ["metrics_summary"]
    lines = [f"\n#### {cfg} ({kind}, 3-seed)  — recorded {datetime.datetime.now().isoformat(timespec='seconds')}\n"]
    # n_params (same across seeds) from first available per-seed json
    npar = None
    per_seed_rows = []
    for s in summ.get("per_seed", []):
        rid = s["run_id"]
        be, sr = "?", "?"
        pj = os.path.join(_RESULTS, f"{rid}.json")
        if os.path.exists(pj):
            with open(pj, encoding="utf-8") as f:
                d = json.load(f)
            be = d.get("best_epoch", "?")
            sr = d.get("stop_reason", "?")
            if npar is None:
                npar = d.get("n_params")
        m = s["metrics"]
        per_seed_rows.append((s["seed"], m.get("aucroc"), m.get("f1"), m.get("mcc"), be, sr))
    if npar is not None:
        lines.append(f"\nn_params: {npar:,} | seeds: {summ.get('seeds')}\n")
    lines.append("\n| metric | mean | std |\n|---|---|---|\n")
    for key, label in GROUP_METRICS:
        if key in ms:
            lines.append(f"| {label} | {ms[key]['mean']:.4f} | {ms[key]['std']:.4f} |\n")
    lines.append("\n| seed | AUCROC | F1 | MCC | best_epoch | stop_reason |\n|---|---|---|---|---|---|\n")
    for seed, au, f1, mcc, be, sr in per_seed_rows:
        au_s = f"{au:.4f}" if isinstance(au, (int, float)) else str(au)
        f1_s = f"{f1:.4f}" if isinstance(f1, (int, float)) else str(f1)
        mc_s = f"{mcc:.4f}" if isinstance(mcc, (int, float)) else str(mcc)
        lines.append(f"| {seed} | {au_s} | {f1_s} | {mc_s} | {be} | {sr} |\n")
    return "".join(lines)


def _secondary_block(cfg, summ):
    ms = summ["metrics_summary"]
    lines = [f"\n#### {cfg} (secondary, 3-seed)  — recorded {datetime.datetime.now().isoformat(timespec='seconds')}\n"]
    lines.append(f"\nseeds: {summ.get('seeds')} | no point-adjustment | per-channel\n")
    lines.append("\n| metric | mean | std |\n|---|---|---|\n")
    for key, label in SECONDARY_METRICS:
        if key in ms:
            lines.append(f"| {label} | {ms[key]['mean']:.4f} | {ms[key]['std']:.4f} |\n")
    lines.append("\n| seed | AUCROC | AUCPR | F1(oracle) |\n|---|---|---|---|\n")
    for s in summ.get("per_seed", []):
        m = s["metrics"]
        lines.append(f"| {s['seed']} | {m.get('aucroc'):.4f} | {m.get('aucpr'):.4f} | {m.get('f1_oracle'):.4f} |\n")
    return "".join(lines)


def write_group(cfg, kind="OPS-SAT"):
    marker = f"#### {cfg} ({kind}, 3-seed)"
    if marker in _read():
        print(f"Group {cfg} already recorded; skipping.")
        return
    summ_path = os.path.join(_RESULTS, f"{cfg}-summary.json")
    if not os.path.exists(summ_path):
        print(f"No summary file for {cfg}; nothing to record.")
        return
    with open(summ_path, encoding="utf-8") as f:
        summ = json.load(f)
    block = _secondary_block(cfg, summ) if kind == "secondary" else _group_block(cfg, summ, kind)
    _append(block)
    print(f"Recorded group {cfg} in GPU_SESSION_LOG.md")


def write_opssat_final():
    if OPSSAT_FINAL_MARKER in _read():
        print("OPS-SAT final section already present; skipping.")
        return

    def L(cfg):
        p = os.path.join(_RESULTS, f"{cfg}-summary.json")
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            return json.load(f)["metrics_summary"]

    out = [f"\n{OPSSAT_FINAL_MARKER}  — {datetime.datetime.now().isoformat(timespec='seconds')}\n",
           "\nAll 48 runs complete (4 models x 4 capacities x 3 seeds), one identical rule, "
           "early-stopped on validation loss (no test-AUCROC selection).\n",
           "\n**AUCROC (3-seed mean +/- std)**\n",
           "\n| capacity | iTransformer | PatchTST | LSTM-AE | AnomalyTransformer |\n|---|---|---|---|---|\n"]
    for c in OPSSAT_CAPS:
        cells = []
        for mm in OPSSAT_MODELS:
            ms = L(f"{mm}-opssat-20260625-d{c}")
            if ms and "aucroc" in ms:
                cells.append(f"{ms['aucroc']['mean']:.4f} +/- {ms['aucroc']['std']:.4f}")
            else:
                cells.append("n/a")
        out.append(f"| d{c} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |\n")

    # d512 headline best_epoch / stop_reason / n_params
    out.append("\n**d512 headline — training detail (per seed)**\n")
    out.append("\n| model | seed | best_epoch | stop_reason | n_params |\n|---|---|---|---|---|\n")
    for mm in OPSSAT_MODELS:
        for s in [42, 0, 1]:
            pj = os.path.join(_RESULTS, f"{mm}-opssat-20260625-d512-seed{s}.json")
            if os.path.exists(pj):
                with open(pj, encoding="utf-8") as f:
                    d = json.load(f)
                out.append(f"| {mm} | {s} | {d.get('best_epoch')} | {d.get('stop_reason')} | {d.get('n_params'):,} |\n")

    out.append("\n**Notes**\n")
    out.append("- iTransformer scales up monotonically (best, 0.964 at d512); PatchTST scales down "
               "(flat-head over-generalization, best at d64); LSTM-AE has a knee at d256 then matches "
               "the transformers; Anomaly Transformer is weakest and unstable.\n")
    out.append("- Anomaly Transformer reaches its lowest validation reconstruction loss at epoch 2-3, "
               "then early-stops (~epoch 17): the minimax objective raises reconstruction val-loss after "
               "the first epochs, so val-loss early stopping keeps a very-early checkpoint. This is a "
               "faithful consequence of the identical rule and explains AT's low scores.\n")
    _append("".join(out))
    print("OPS-SAT final consolidated section appended.")


ANALYSES_MARKER = "### Step 4-5 analyses + secondary — FINAL (20260625)"


def _safe_json(name):
    p = os.path.join(_RESULTS, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def write_analyses_final():
    if ANALYSES_MARKER in _read():
        print("Analyses-final section already present; skipping.")
        return
    out = [f"\n{ANALYSES_MARKER}  - {datetime.datetime.now().isoformat(timespec='seconds')}\n"]

    # Secondary 3-seed table
    out.append("\n**Secondary validation (d=512, 3-seed mean +/- std, no point-adjustment)**\n")
    out.append("\n| model | dataset | AUCROC | AUCPR | F1(oracle) |\n|---|---|---|---|---|\n")
    for m in ["patchtst", "itransformer", "anomaly-transformer"]:
        for ds in ["smap", "msl"]:
            d = _safe_json(f"{m}-{ds}-20260625-summary.json")
            if not d:
                continue
            s = d["metrics_summary"]
            out.append(f"| {m} | {ds} | {s['aucroc']['mean']:.4f} +/- {s['aucroc']['std']:.4f} | "
                       f"{s['aucpr']['mean']:.4f} +/- {s['aucpr']['std']:.4f} | "
                       f"{s['f1_oracle']['mean']:.4f} +/- {s['f1_oracle']['std']:.4f} |\n")

    # SHAP concentration (d512)
    shap = _safe_json("shap_concentration_20260625.json")
    if shap:
        out.append("\n**SHAP temporal concentration (d=512, seed 42, 20 shared TP segments)**\n")
        out.append("\n| model | concentration (top region) | ratio | per-seg max attribution |\n|---|---|---|---|\n")
        for key in ["at", "patchtst", "itransformer"]:
            if key in shap:
                e = shap[key]
                out.append(f"| {e.get('label', key)} | {e['mean_conc']*100:.1f}% +/- {e['std_conc']*100:.1f}% | "
                           f"{e['ratio']:.2f}x | {e['mean_seg_max']:.4f} +/- {e['std_seg_max']:.4f} |\n")

    # AT scoring variants
    atv = _safe_json("at-scoring-variants-20260625.json")
    if atv:
        out.append("\n**Anomaly Transformer scoring-variant comparison (d=512, seed 42, no retraining)**\n")
        out.append("\n| variant | AUCROC | MCC |\n|---|---|---|\n")
        for v in ["composite", "assdis", "mse"]:
            if v in atv["variants"]:
                e = atv["variants"][v]
                tag = " (adopted)" if v == "mse" else ""
                out.append(f"| {v}{tag} | {e['aucroc']:.4f} | {e['mcc']:.4f} |\n")

    # Ensemble
    ens = _safe_json("ensemble-opssat-20260625-summary.json")
    if ens:
        s = ens["metrics_summary"]
        out.append("\n**Ensemble (z-score-standardized score averaging, 3 transformers x 3 seeds)**\n")
        out.append(f"\nAUCROC {s['aucroc']['mean']:.4f} +/- {s['aucroc']['std']:.4f} | "
                   f"F1 {s['f1']['mean']:.4f} +/- {s['f1']['std']:.4f} | "
                   f"MCC {s['mcc']['mean']:.4f} +/- {s['mcc']['std']:.4f} | "
                   f"AUCPR {s['aucpr']['mean']:.4f} +/- {s['aucpr']['std']:.4f}\n")

    # Inference benchmark
    inf = _safe_json("inference-benchmark-20260625.json")
    if inf:
        out.append(f"\n**Inference benchmark ({inf.get('gpu_name','?')})**\n")
        out.append("\n| model | latency ms/seg (batch=1) | throughput seg/s (batch=64) |\n|---|---|---|\n")
        for m, e in inf.get("models", {}).items():
            out.append(f"| {m} | {e['latency_ms_per_segment_mean']:.2f} +/- {e['latency_ms_per_segment_std']:.2f} | "
                       f"{e['throughput_segments_per_sec']:.0f} |\n")

    # Figures + sources
    out.append("\n**Figures regenerated (papers/satellite-anomaly/figures/)**\n")
    out.append("- metrics_comparison, roc_pr_curves, score_distributions: 4 models at d=512 "
               "(*-opssat-20260625-d512 summaries + seed42 scores).\n")
    out.append("- secondary_validation, cross_dataset_aucroc: 3 transformers, 20260625 d=512 + secondary summaries.\n")
    out.append("- per_channel_f1: d=512 seed-42 score NPZs (4 models).\n")
    out.append("- capacity_ablation: AUCROC vs {64,128,256,512}, all 4 models, 3-seed mean +/- std.\n")
    out.append("- shap_*_overlay, shap_mean_attribution: GradientSHAP on d=512 seed-42 checkpoints.\n")
    out.append("- attention_heatmap_tp, attention_tp_vs_tn: PatchTST d=512 seed-42 checkpoint "
               "(patchtst-opssat-20260625-d512-seed42-best.pt).\n")
    _append("".join(out))
    print("Analyses-final section appended.")


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: append_manifest.py --header | --group <cfg> | --secondary <cfg> | --opssat-final | --analyses-final")
        sys.exit(1)
    if args[0] == "--header":
        write_header()
    elif args[0] == "--group" and len(args) > 1:
        write_group(args[1], kind="OPS-SAT")
    elif args[0] == "--secondary" and len(args) > 1:
        write_group(args[1], kind="secondary")
    elif args[0] == "--opssat-final":
        write_opssat_final()
    elif args[0] == "--analyses-final":
        write_analyses_final()
    else:
        print("Bad arguments.")
        sys.exit(1)


if __name__ == "__main__":
    main()
