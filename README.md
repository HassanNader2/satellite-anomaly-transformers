# satellite-anomaly-transformers

Experiment code for:  
**Satellite Telemetry Anomaly Detection Using Transformer-Based Machine Learning:
A Benchmark Study with Interpretability**  
Hassan Nader, 2026

---

## What this repo contains

- `src/` -- training, evaluation, SHAP analysis, and per-channel breakdown scripts
- `configs/` -- YAML configs for all model runs (PatchTST, iTransformer, Anomaly Transformer, LSTM-AE)
- `results/` -- per-seed and summary JSON files for all reported metrics
- Model wrappers live in `src/models/`

## Quick start

**Requirements:** Python 3.11, CPU-only (no CUDA/MPS needed)

```bash
pip install -r requirements.txt
```

Clone the three model repos into `models/`:

```bash
mkdir -p models
git clone --depth 1 https://github.com/yuqinie98/PatchTST models/PatchTST
git clone --depth 1 https://github.com/thuml/Anomaly-Transformer models/Anomaly-Transformer
git clone --depth 1 https://github.com/thuml/iTransformer models/iTransformer
```

Download the OPS-SAT-AD dataset from Zenodo DOI 10.5281/zenodo.12588359 and place
`segments.csv` and `dataset.csv` in `../data/`.

**Run one model (example):**

```bash
python src/run_experiment.py patchtst patchtst-opssat --seed 42
python src/run_experiment.py itransformer itransformer-opssat --seed 42
python src/run_experiment.py anomaly_transformer anomaly-transformer-opssat --seed 42
python src/run_experiment.py lstm_ae lstm-ae-opssat --seed 42
```

**Summarize across seeds:**

```bash
python src/summarize_results.py patchtst-opssat-20260516-01
```

**SHAP analysis:**

```bash
python src/shap_analysis.py
```

**Per-channel breakdown:**

```bash
python src/analyze_per_channel.py
```

## Hardware note

All reported results were produced on an Intel Mac (CPU-only, no GPU).
Floating-point results are reproducible within approximately +-0.001 of reported
metrics across identical seeds. Minor variation is expected due to thread ordering.

Training times per seed: PatchTST ~20 min, iTransformer ~15 min,
Anomaly Transformer ~150 min, LSTM-AE ~30 min.

## License

MIT License. See LICENSE.
