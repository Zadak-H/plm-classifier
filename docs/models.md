# Models & pLMs

Run `plm-classifier list` to see what is installed/available in your environment.

## Classifiers

`kind` = classical (sklearn / xgboost / lightgbm) or torch (deep). `max rows` is the dataset-size
ceiling the size engine uses to auto-skip models that would be too slow. All are probability-capable.

| model | kind | max rows | notes |
|-------|------|----------|-------|
| `logreg` | classical | unlimited | logistic regression |
| `sgd` | classical | unlimited | scalable linear (log/ modified-huber) |
| `lda` | classical | unlimited | linear discriminant |
| `gnb` | classical | unlimited | Gaussian naive Bayes |
| `qda` | classical | ≤ 200,000 | quadratic discriminant |
| `knn` | classical | ≤ 50,000 | |
| `svc_rbf` | classical | ≤ 20,000 | probability=True |
| `gpc` | classical | ≤ 2,000 | Gaussian process |
| `rf` | classical | ≤ 200,000 | |
| `extra_trees` | classical | ≤ 200,000 | |
| `hist_gb` | classical | unlimited | |
| `xgboost` | classical | unlimited | optional |
| `lightgbm` | classical | unlimited | optional |
| `mlp` | classical | ≤ 50,000 | sklearn MLP |
| `mlp_torch` | torch | unlimited | FNN over embeddings (+tabular) |
| `cnn1d` | torch | unlimited | 1D-CNN; one-hot/blosum only |

Class imbalance: tree/linear/SVC models search a `class_weight` (`balanced`) option automatically.

## Protein language models

Same menu as PLM-Regressor: ESM2 (8M–15B), ESM1/1b/1v, ESM C (`esmc_300m/600m`) and the HF mirror
ESM++ (`esmplusplus_small/large`), ProtT5 (xl/half/bfd/xxl), ProstT5, ProtBert(/bfd), Ankh
base/large, ProSST, CARP.

!!! note "ESM C"
    `esmc_*` need EvolutionaryScale's `esm` SDK (collides with `fair-esm`). Use
    `esmplusplus_small` / `esmplusplus_large` for the same embeddings via `transformers`.

## Simple encodings + tabular

- `onehot`, `blosum62` — positional encodings (aligned, equal-length sequences); the only inputs
  `cnn1d` accepts.
- `tabular` — your extra numeric/categorical columns, combined with any other source.
