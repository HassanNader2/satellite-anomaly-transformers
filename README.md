# Capacity-Matched Benchmark of Deep Anomaly Detection on Real Satellite Telemetry

Code accompanying the paper *"Capacity Rivals Architecture: A Capacity-Matched Benchmark of Deep
Anomaly Detection on Real Satellite Telemetry with Dual-Layer Explainability."*

It benchmarks four reconstruction-based anomaly detectors, PatchTST, iTransformer, Anomaly
Transformer, and an LSTM autoencoder, on the ESA OPS-SAT-AD dataset under one identical training
rule across four matched capacities and three seeds, with secondary validation on NASA SMAP/MSL
(non-point-adjusted) and dual-layer explainability (GradientSHAP + attention).

## Repository layout

```
src/                  training, evaluation, analysis, and explainability code
  models/             thin wrappers around the four architectures
configs/              YAML configs (one per model x capacity); *-20260625-d{64,128,256,512} are the paper runs
requirements.txt      Python dependencies
```

The scripts expect the following sibling directories, created by you (see below):
`models/` (third-party architecture code), `data/`, `results/`, `logs/`, `checkpoints/`.

## Environment

- Python 3.11 (developed on 3.11.9)
- GPU used in the paper: NVIDIA GeForce RTX 4060 (8 GB), CUDA 12.1, PyTorch 2.5.1. Runs on CPU too.

```
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
# For CUDA, install the matching torch build, e.g.:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Third-party architecture code (clone into ./models/)

The wrappers in `src/models/` import the original author implementations. Clone them into a
`models/` directory next to `src/` and `configs/`:

```
mkdir models && cd models
git clone https://github.com/yuqinie98/PatchTST.git PatchTST
git clone https://github.com/thuml/iTransformer.git iTransformer
git clone https://github.com/thuml/Anomaly-Transformer.git Anomaly-Transformer
cd ..
```

(The LSTM autoencoder is implemented directly in `src/models/lstm_ae_wrapper.py`.)

## Datasets

- **OPS-SAT-AD** (primary): Zenodo DOI 10.5281/zenodo.12588359. Place `segments.csv` and
  `dataset.csv` under `data/`.
- **SMAP / MSL** (secondary): original Hundman et al. release, https://github.com/khundman/telemanom
  (or Kaggle `patrickfleith/nasa-anomaly-detection-dataset-smap-msl`). Place per-channel `.npy`
  files under `data/smap/` and `data/msl/`.

## Training protocol (identical rule for every run)

All models share: AdamW (lr 1e-4), batch 16, gradient clipping (max-norm 1.0), ReduceLROnPlateau
(factor 0.5, patience 5, floor 1e-6), early stopping (max 150 epochs, patience 15, min_delta 1e-4
relative on validation reconstruction MSE, best checkpoint restored). Architectural capacity is
identical per run (d_model / hidden size, n_heads=8, e_layers=3, d_ff=d_model). Capacity is swept
over {64, 128, 256, 512}. The anomaly threshold is a 200-point best-F1 sweep on the validation set,
applied unchanged to the test set. Training is unsupervised; epochs are never tuned to test scores.

## Reproducing the paper

OPS-SAT-AD, one model at one capacity and seed (config name = YAML filename without `.yaml`):

```
python src/run_experiment.py patchtst            patchtst-opssat-20260625-d512            --seed 42
python src/run_experiment.py itransformer        itransformer-opssat-20260625-d512        --seed 42
python src/run_experiment.py anomaly-transformer anomaly-transformer-opssat-20260625-d512 --seed 42
python src/run_experiment.py lstm-ae             lstm-ae-opssat-20260625-d512             --seed 42
```

Run each model at capacities d64/d128/d256/d512 and seeds 42/0/1, then aggregate to mean +/- std:

```
python src/summarize_results.py patchtst-opssat-20260625-d512     # repeat per model x capacity
```

Secondary validation (SMAP/MSL, 3 Transformers, non-point-adjusted):

```
python src/run_secondary.py --dataset smap --model patchtst --seed 42
python src/run_secondary.py --dataset msl  --model itransformer --seed 0
python src/summarize_secondary.py
```

Analysis and explainability:

```
python src/select_shap_samples.py        # shared TP/TN segments on the d=512 seed-42 checkpoints
python src/shap_analysis.py              # GradientSHAP for the three Transformers
python src/attention_viz.py              # PatchTST attention entropy + heatmaps
python src/at_scoring_variants.py        # AT composite vs. discrepancy vs. masked-MSE scoring
python src/compute_ensemble.py           # z-score-standardised 3-model ensemble
python src/benchmark_inference.py        # forward-pass latency / throughput
python src/analyze_results.py            # main comparison figures
python src/analyze_per_channel.py        # per-channel breakdown table/figure
python src/plot_results.py               # capacity-ablation figure
```

Outputs are written to `results/` (JSON + score `.npz`), `logs/`, `checkpoints/`, and figures to
`../figures/` (adjust paths as needed).

## Notes

- Results are deterministic in seed but not bit-identical across hardware
  (`torch.use_deterministic_algorithms` is not enabled).
- iTransformer at d_model=512 did not plateau within the 150-epoch budget and is reported as a
  lower bound in the paper.

## License

MIT (see LICENSE).
