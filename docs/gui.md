# GUI usage

```bash
plm-classifier gui          # or: streamlit run app.py
```

Opens at `http://localhost:8501`. Seven tabs, left to right.

```
┌ PLM-Classifier ─────────────────────────────────────────────────────────────┐
│ 1.Data │ 2.Features │ 3.Models │ 4.Search │ 5.Run │ 6.Results │ 7.Predict │
└───────────────────────────────────────────────────────────────────────────────┘
```

## 1 · Data

- Upload your training **CSV**.
- Pick the **sequence column** and the **label column** (binary `yes`/`no`, `0`/`1`, or any set
  of class names).
- Pick the **positive / ranking class** — candidates are later ranked by this class's probability
  (default: the last class alphabetically).
- Optional: **ID column**, **group column** (leakage-safe stratified CV), and **extra inputs**
  (numeric like pH/temperature, and/or categorical like buffer).

## 2 · Features

Multiselect feature sources (pLMs, `onehot`/`blosum62`, `tabular`). Set the embedding directory;
extract any missing pLM bank with one click (cached afterwards).

## 3 · Models

Multiselect classifiers (classical + deep `mlp_torch`/`cnn1d`). Models too costly for the dataset
size are auto-skipped.

## 4 · Search

- **Primary metric**: `roc_auc | pr_auc | f1 | f1_macro | accuracy | balanced_accuracy | mcc | log_loss`.
- **Auto-tune by dataset size** (recommended) chooses stratified CV, trial budget, and gating.
- Toggle uncertainty (ensemble probability spread) and ensemble size.

## 5 · Run

Set the output directory and click **🚀 Run training**; a live log streams the Optuna search.

## 6 · Results

Best model, classes, the **OOF metric table**, and plots: **confusion matrix, ROC, precision-recall,
calibration**, and a per-model comparison bar. Download `best_model.joblib`, `oof_predictions.csv`,
and the candidate ranking.

## 7 · Predict

Point at a saved run, upload a **candidate CSV**, set the sequence column (and optionally a class to
rank by), and **Rank candidates** → ranked table + `top_{10,50,100}.csv` by class probability.

!!! tip "Reproducible"
    Each run writes `run_config.yaml`; re-run headless with `plm-classifier train run_config.yaml`.
