# PLM-Classifier

> **General sequence → class label classification for protein engineering.** Predict a
> yes/no outcome or assign one of many classes from a sequence (+ optional extra columns) —
> with a no-code web GUI.

[Get started](install.md){ .md-button .md-button--primary }
[GUI usage](gui.md){ .md-button }
[GitHub](https://github.com/Zadak-H/plm-classifier){ .md-button }

The classification sibling of [PLM-Regressor](https://github.com/Zadak-H/plm-regressor) — same
engine, classification head. Binary and multiclass both supported.

## What it does

- **Many feature sources** — protein language model embeddings (ESM2 8M–15B, ESM1/1b/1v,
  ESM C / ESM++, ProtT5, ProstT5, ProtBert, ProSST, Ankh, CARP), simple encodings
  (one-hot, BLOSUM62), and **extra tabular columns** (pH, temperature, assay conditions).
- **Many classifiers** — classical (LogReg, SVC, KNN, RF, ExtraTrees, HistGB, XGBoost,
  LightGBM, LDA/QDA, GaussianNB, GPC, SGD) **plus deep models**: `mlp_torch` (FNN over
  embeddings) and `cnn1d` (1D-CNN over one-hot/BLOSUM).
- **Size-aware Optuna search** — auto-picks **stratified** CV strategy, trial budget, and
  eligible models from the dataset size (100 → 1M+).
- **OOF model selection**, probability outputs, ensemble/entropy **uncertainty**, and a rich
  report (accuracy, balanced accuracy, F1, ROC-AUC, PR-AUC, MCC, log-loss + confusion/ROC/PR
  plots).
- **Streamlit GUI** + a thin CLI; ranks candidates by the probability of your chosen class.

## 60-second tour

```bash
pip install -e ".[all]"     # or pick extras: .[deep] .[esm] .[t5] .[gui]
plm-classifier gui          # upload CSV → pick label/features/models → Run → rank
```

```bash
plm-classifier train run.yaml
plm-classifier predict --run-dir runs/binders --candidate-csv cand.csv --predict-seq-col Sequence
```

## How model selection scales with data size

| rows | CV (stratified) | tuned models |
|------|-----------------|--------------|
| <300 | RepeatedStratifiedKFold 5×3 | all |
| 300–1k | StratifiedKFold 5 (group-aware) | all |
| 1k–5k | StratifiedKFold 5 | all but exact GPC |
| 5k–50k | StratifiedKFold 3 / holdout | drop GPC/SVC-rbf |
| 50k–500k | stratified holdout | scalable only (SGD/LogReg, HistGB/XGB/LGB, torch) |
| 500k–1M+ | stratified holdout | scalable only (mmap embeddings) |

Full classifier + pLM list: [Models & pLMs](models.md).
