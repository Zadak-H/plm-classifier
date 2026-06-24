<h1 align="center">PLM-Classifier</h1>

<p align="center">
  <b>General sequence → class label classification for protein engineering.</b><br>
  Predict yes/no or assign one of many classes from a sequence (+ optional extra columns) —
  with a no-code web GUI. The classification sibling of PLM-Regressor.
</p>

<p align="center">
  <a href="https://zadak-h.github.io/plm-classifier/"><img src="https://img.shields.io/badge/docs-website-indigo.svg" alt="Docs"></a>
  <a href="requirements.txt"><img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
</p>

<p align="center">
  📖 <b><a href="https://zadak-h.github.io/plm-classifier/">Documentation &amp; GUI guide</a></b>
</p>

---

## What it does

- **Binary (yes/no) and multiclass** classification from protein sequences.
- **Many feature sources** — pLM embeddings (ESM2 8M–15B, ESM1/1b/1v, ESM C / ESM++, ProtT5,
  ProstT5, ProtBert, ProSST, Ankh, CARP), one-hot/BLOSUM62, and **extra tabular columns**
  (pH, temperature, assay conditions).
- **Many classifiers** — classical (LogReg, SVC, KNN, RF, ExtraTrees, HistGB, XGBoost, LightGBM,
  LDA/QDA, GaussianNB, GPC, SGD) **plus deep** `mlp_torch` (FNN) and `cnn1d` (1D-CNN).
- **Size-aware, stratified Optuna search** (100 → 1M+), OOF model selection, probability outputs,
  ensemble/entropy uncertainty.
- Metrics: accuracy, balanced accuracy, F1, ROC-AUC, PR-AUC, MCC, log-loss + confusion/ROC/PR/
  calibration plots.
- **Streamlit GUI** + thin CLI; ranks candidates by the probability of your chosen class.

## Install

```bash
git clone https://github.com/Zadak-H/plm-classifier.git
cd plm-classifier
python -m pip install -e ".[all]"      # or extras: .[deep] .[esm] .[t5] .[gui]
```

## Quick start

```bash
plm-classifier gui                      # no-code
# or
plm-classifier train run.yaml
plm-classifier predict --run-dir runs/binders --candidate-csv cand.csv --predict-seq-col Sequence
```

```yaml
# run.yaml
csv: data/binders.csv
seq_col: Protein_Seq
target_col: label            # yes/no, 0/1, or class names
positive_class: "yes"
embedding_dir: embeddings
feature_sources: [esm2, tabular]
extra_feature_cols: [pH, temp]
models: [logreg, svc_rbf, hist_gb, mlp_torch]
metric: roc_auc
auto_size: true
out_dir: runs/binders
```

## Documentation

**<https://zadak-h.github.io/plm-classifier/>** (source in [`docs/`](docs/)).

## Repository layout

- `plm_classifier/` — framework package (config, registry, sizing, features, models, search, train,
  predict, metrics, plots, report, cli) + `plm_classifier/embeddings/` extractors
- `app.py` — Streamlit GUI
- `docs/` — documentation site (MkDocs Material)
- `data/`, `embeddings/` — example datasets and precomputed embedding banks

## License

[MIT](LICENSE).
