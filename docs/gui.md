# GUI usage

The Streamlit GUI is the no-code path: upload data, pick a language model, train, and rank
candidates — no Python knowledge required.

```bash
plm-classifier gui          # or: streamlit run app.py
```

Opens at `http://localhost:8501`. Eight tabs, left to right:

```
┌ PLM-Classifier ─────────────────────────────────────────────────────────────────────┐
│ Embed │ Data │ Features │ Models │ Search │ Run │ Results │ Predict │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 0 · Embed — generate embeddings without training

!!! tip "Start here if you just need embedding files"
    This tab works completely standalone. No training data, no labels needed.
    Upload any CSV of sequences → pick a pLM → click **Generate embeddings** → download the `.npz`.

- Upload a **CSV** with a column of protein sequences (any column name).
- Pick the **sequence column**.
- Choose a **protein language model** from the dropdown — every model shows a plain-English
  description with size and speed information. Default is **ESM-2 650M**.
- *Advanced options* (collapsed): batch size and CPU-only mode.
- Click **Generate embeddings**. The first run downloads model weights (~minutes).
- Click **Download `<plm>.npz`** when done.

!!! note "Using the file for training"
    Put the `.npz` in your embedding directory (default `embeddings/`) and select that pLM
    in the **Features** tab.

### Which model to pick?

| Model | Speed | Memory | When to use |
|-------|-------|---------|-------------|
| `esm2_8m` | ⚡⚡⚡ | ~200 MB | Quick prototyping, CPU only |
| `esm2_35m` | ⚡⚡⚡ | ~300 MB | Good starting point on CPU |
| `esm2_150m` | ⚡⚡ | ~600 MB | Balanced quality/speed |
| `esm2` (650M) | ⚡⚡ | ~2.5 GB | **Recommended default** — strong across benchmarks |
| `esmc_300m` | ⚡⚡ | ~1.5 GB | Latest ESM-C; auto-detected and runs transparently |
| `esmc_600m` | ⚡ | ~3 GB | Larger ESM-C |
| `protT5` | ⚡ | ~5 GB | Strong T5-based alternative |
| `ankh_base` | ⚡⚡ | ~400 MB | Compact, efficient |

### ESM-C (esmc_300m / esmc_600m)

!!! success "ESM-C just works"
    The tool automatically finds a conda environment on your machine that has the ESM SDK
    installed and uses it transparently. You'll see `[ESM-C] using: /path/to/env/python3`
    in the log.

To set it up once:

```bash
conda create -n esmc python=3.10 -y
conda activate esmc
pip install esm httpx
```

After that, pick `esmc_300m` or `esmc_600m` anywhere in the GUI or CLI — it just works.

---

## 1 · Data

- Upload your training **CSV**.
- Pick the **sequence column** and the **label column** (binary `yes`/`no`, `0`/`1`, or any set
  of class names).
- Pick the **positive / ranking class** — candidates are ranked by this class's probability.
- Optional: **ID column**, **group column** (leakage-safe stratified CV), and **extra inputs**
  (numeric like pH/temperature, categorical like buffer).

---

## 2 · Features

Multiselect feature sources (pLMs, `onehot`/`blosum62`, `tabular`). Set the embedding directory;
extract any missing pLM bank with one click (cached afterwards).

!!! tip "Already generated embeddings in the Embed tab?"
    Place the `.npz` in the embedding directory, then select the matching pLM here.

---

## 3 · Models

Multiselect classifiers (classical + deep `mlp_torch`/`cnn1d`). Models too costly for the dataset
size are auto-skipped.

---

## 4 · Search

- **Primary metric**: `roc_auc | pr_auc | f1 | f1_macro | accuracy | balanced_accuracy | mcc | log_loss`.
- **Auto-tune by dataset size** (recommended) chooses stratified CV, trial budget, and gating.
- Toggle uncertainty (ensemble probability spread) and ensemble size.

---

## 5 · Run

Set the output directory and click **🚀 Run training**; a live log streams the Optuna search.

---

## 6 · Results

Best model, classes, the **OOF metric table**, and plots: **confusion matrix, ROC, precision-recall,
calibration**, and a per-model comparison bar. Download `best_model.joblib`, `oof_predictions.csv`,
and the candidate ranking.

---

## 7 · Predict

- Point at a saved run, upload a **candidate CSV** — the sequence column is auto-detected from
  the file.
- The **embedding directory** defaults to the one used during training.
- Optionally override the class to rank by, set **Top N**, and click **Rank candidates**.

!!! warning "Missing embeddings"
    If candidates don't have embeddings yet, the tool shows a clear warning. Use the **Embed
    tab (Tab 0)** to generate them first, then point the candidate embedding directory there.

!!! tip "Reproducible"
    Each run writes `run_config.yaml`; re-run headless with `plm-classifier train run_config.yaml`.
