# How it works

Two-stage, probability-first classification.

```
sequence ──► frozen protein LM ──► pooled embedding ─┐
(+ extra columns) ──────────────► tabular features ──┼─► preprocessing ─► classifier ─► class probabilities
(one-hot / blosum) ─────────────► positional enc. ───┘        (Optuna-searched pipeline)
```

The pretrained pLM is a **frozen feature extractor**; only a light classifier head is trained.

## Labels & aggregation

The label column may be `yes`/`no`, `0`/`1`, or any class names; labels are encoded to `0..K-1`
(sorted for determinism) and the original names are kept for reporting. Repeated rows for the same
sequence are reduced by **majority vote** (`majority_by_sequence`) unless you choose `keep_rows`.

## Search

A single Optuna **TPE** study jointly tunes feature subset, feature mode, classifier family,
preprocessing (variance filter, scaler, `f_classif`/mutual-info selection, PCA/SVD), and
per-model hyperparameters (including `class_weight` for imbalance). The CNN is constrained to
positional encodings (invalid combinations are pruned).

## Size-aware, stratified CV

Row count selects a profile that sets the **stratified** CV strategy, trial budget, and eligible
models (kernel methods like GPC/SVC-rbf are dropped on large data; SGD/linear, boosting, and
mini-batch torch scale up). Big data tunes on a class-balanced **subsample**, then the best config
is **refit on the full dataset**.

## Selection, metrics & uncertainty

Selection uses **out-of-fold probabilities** (never training fit). Metrics: accuracy, balanced
accuracy, F1 (binary positive-class and macro), precision/recall, ROC-AUC, PR-AUC, MCC, log-loss
(binary directly; multiclass via one-vs-rest macro). Uncertainty (optional) = spread of class
probabilities across the top-N ensemble, plus per-row predictive **entropy** and **confidence**
(max probability).

## Deployment

`best_model.joblib` (a `FittedRunModel` with the estimator, class names, and feature spec) drives
`plm-classifier predict`, which outputs each candidate's predicted class + per-class probabilities
and ranks by the probability of your chosen **positive class** (`top_{10,50,100}.csv`).
