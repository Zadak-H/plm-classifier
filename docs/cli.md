# CLI usage

```bash
plm-classifier train   run.yaml
plm-classifier predict --run-dir runs/X --candidate-csv cand.csv --predict-seq-col Sequence
plm-classifier embed   --plm esm2 --input-csv X.csv --seq-col Sequence --output-npz embeddings/esm2.npz
plm-classifier list
```

## Train from a config

```yaml
# run.yaml
csv: data/binders.csv
seq_col: Protein_Seq
target_col: label            # yes/no, 0/1, or any class names
positive_class: "yes"        # which class to rank candidates by (optional)
embedding_dir: embeddings

feature_sources: [esm2, tabular]
extra_feature_cols: [pH, temp]
categorical_cols: [buffer]

models: [logreg, svc_rbf, hist_gb, mlp_torch]
metric: roc_auc              # roc_auc|pr_auc|f1|f1_macro|accuracy|balanced_accuracy|mcc|log_loss
auto_size: true
out_dir: runs/binders
```

```bash
plm-classifier train run.yaml
```

Useful keys: `standard_search`, `no_uncertainty`, `top_ensemble`, `n_trials`, `group_col`,
`wt_sequence` (delta feature modes), `use_gpu`, `replicate_policy` (`majority_by_sequence` or
`keep_rows`).

## Rank candidates

```bash
plm-classifier predict --run-dir runs/binders \
  --candidate-csv data/candidates.csv --predict-seq-col Sequence \
  --candidate-embedding-dir zeroshot_embeds --positive-class yes --top-n 100
```

Writes `candidate_predictions.csv` (with `pred_label`, per-class `proba_*`, `confidence`, `entropy`,
`score` = P(positive class)) + `top_{10,50,100}.csv`.

## Outputs per run

`best_model.joblib`, `oof_predictions.csv`, `train_predictions.csv`, `search_history.csv`,
`fold_metrics.csv`, `run_summary.json`, `coverage_report.json`, `run_config.yaml`,
`run_report.json` + `run_report.html` (+ `plots/`). With uncertainty:
`uncertainty_ensemble.joblib`, `top_ensemble_members.csv`. With tabular features: `tabular_encoder.joblib`.
