"""
Select the fixed test-set sample indices for SHAP and attention visualization.

Strategy:
- Use seed 42 checkpoints and thresholds for all three models.
- Select segments classified correctly by ALL THREE models:
    - 20 true positives  (anomalous, all three predicted anomalous)
    - 20 true negatives  (nominal, all three predicted nominal)
- Additionally record per-model FP and FN sets (up to 10 each) for failure analysis.
- Save to experiments/results/shap_sample_indices.json

Run from project root with venv active:
  python papers/satellite-anomaly/experiments/src/select_shap_samples.py
"""

import os
import json
import numpy as np

_HERE      = os.path.dirname(os.path.abspath(__file__))
_RESULTS   = os.path.join(_HERE, "..", "results")

MODELS = [
    {
        "key":         "at",
        "label":       "Anomaly Transformer",
        "scores_path": os.path.join(_RESULTS, "anomaly-transformer-opssat-20260516-d64-01-seed42-scores.npz"),
        "json_path":   os.path.join(_RESULTS, "anomaly-transformer-opssat-20260516-d64-01-seed42.json"),
    },
    {
        "key":         "patchtst",
        "label":       "PatchTST",
        "scores_path": os.path.join(_RESULTS, "patchtst-opssat-20260516-01-seed42-scores.npz"),
        "json_path":   os.path.join(_RESULTS, "patchtst-opssat-20260516-01-seed42.json"),
    },
    {
        "key":         "itransformer",
        "label":       "iTransformer",
        "scores_path": os.path.join(_RESULTS, "itransformer-opssat-20260516-01-seed42-scores.npz"),
        "json_path":   os.path.join(_RESULTS, "itransformer-opssat-20260516-01-seed42.json"),
    },
]

N_SHARED   = 20   # TP and TN samples that all three models agree on
N_FAILURES = 10   # per-model FP and FN samples


def load_model(m):
    data = np.load(m["scores_path"])
    scores = data["test_scores"]
    labels = data["test_labels"].astype(int)
    with open(m["json_path"]) as f:
        meta = json.load(f)
    threshold = meta["metrics"]["threshold"]
    preds = (scores >= threshold).astype(int)
    return scores, labels, preds, threshold


def main():
    rng = np.random.default_rng(42)

    results = {}
    labels_ref = None
    pred_list  = []
    score_list = []

    for m in MODELS:
        scores, labels, preds, threshold = load_model(m)

        if labels_ref is None:
            labels_ref = labels
        else:
            assert np.array_equal(labels_ref, labels), "Label mismatch across models"

        pred_list.append(preds)
        score_list.append(scores)

        tp = np.where((labels == 1) & (preds == 1))[0]
        tn = np.where((labels == 0) & (preds == 0))[0]
        fp = np.where((labels == 0) & (preds == 1))[0]
        fn = np.where((labels == 1) & (preds == 0))[0]

        results[m["key"]] = {
            "label":     m["label"],
            "threshold": float(threshold),
            "tp_count":  int(len(tp)),
            "tn_count":  int(len(tn)),
            "fp_count":  int(len(fp)),
            "fn_count":  int(len(fn)),
            "fp_indices": rng.choice(fp, size=min(N_FAILURES, len(fp)), replace=False).tolist() if len(fp) else [],
            "fn_indices": rng.choice(fn, size=min(N_FAILURES, len(fn)), replace=False).tolist() if len(fn) else [],
        }

        print(f"{m['label']:25s}  TP={len(tp):3d}  TN={len(tn):3d}  FP={len(fp):3d}  FN={len(fn):3d}")

    # --- Shared correct predictions ---
    all_preds = np.stack(pred_list, axis=0)   # (3, N_test)

    # Segments all three got right, by true class
    agreed_tp_mask = (labels_ref == 1) & np.all(all_preds == 1, axis=0)
    agreed_tn_mask = (labels_ref == 0) & np.all(all_preds == 0, axis=0)

    agreed_tp = np.where(agreed_tp_mask)[0]
    agreed_tn = np.where(agreed_tn_mask)[0]

    print(f"\nShared TPs (all 3 correct): {len(agreed_tp)}")
    print(f"Shared TNs (all 3 correct): {len(agreed_tn)}")

    n_tp = min(N_SHARED, len(agreed_tp))
    n_tn = min(N_SHARED, len(agreed_tn))

    # Sort by descending mean score for anomalous (clearest positives first),
    # ascending mean score for nominal (clearest negatives first).
    mean_scores = np.mean(np.stack(score_list, axis=0), axis=0)

    tp_sorted = agreed_tp[np.argsort(-mean_scores[agreed_tp])][:n_tp]
    tn_sorted = agreed_tn[np.argsort( mean_scores[agreed_tn])][:n_tn]

    shared_tp = tp_sorted.tolist()
    shared_tn = tn_sorted.tolist()

    print(f"\nSelected shared TPs: {len(shared_tp)}")
    print(f"Selected shared TNs: {len(shared_tn)}")

    # --- Build output ---
    output = {
        "description": (
            "Fixed test-set indices for SHAP and attention visualization. "
            "shared_tp: anomalous segments all 3 models correctly flagged (sorted by descending mean score). "
            "shared_tn: nominal segments all 3 models correctly cleared (sorted by ascending mean score). "
            "per_model fp/fn: failure-mode samples for each model individually."
        ),
        "seed":        42,
        "n_test":      int(len(labels_ref)),
        "shared_tp":   shared_tp,
        "shared_tn":   shared_tn,
        "per_model":   results,
    }

    out_path = os.path.join(_RESULTS, "shap_sample_indices.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

    # --- Summary table ---
    print("\n--- Mean anomaly scores for selected TPs ---")
    for m, scores in zip(MODELS, score_list):
        tp_scores = scores[shared_tp]
        tn_scores = scores[shared_tn]
        print(f"  {m['label']:25s}  TP mean={tp_scores.mean():.4f}  TN mean={tn_scores.mean():.4f}")


if __name__ == "__main__":
    main()
