# Install

Requires Python 3.9+.

```bash
git clone https://github.com/Zadak-H/plm-classifier.git
cd plm-classifier
python -m pip install -e ".[all]"
```

Pick smaller extras instead of `.[all]`:

| extra | brings | for |
|-------|--------|-----|
| (core) | numpy, pandas, scikit-learn, scipy, optuna, matplotlib, pyyaml | always |
| `.[deep]` | torch | `mlp_torch`, `cnn1d` |
| `.[boost]` | xgboost, lightgbm | gradient boosting |
| `.[esm]` | fair-esm | ESM2 / ESM1 embeddings |
| `.[t5]` | transformers, sentencepiece | ProtT5 / ProstT5 / ProtBert / Ankh / ESM++ |
| `.[gui]` | streamlit | the web GUI |

## Verify

```bash
plm-classifier list      # classifiers + pLMs and whether each is available
plm-classifier gui       # launch the web app
```

## Notes

- **ESM C native** (`esmc_*`) needs EvolutionaryScale's `esm` SDK, which collides with `fair-esm`;
  use **ESM++** (`esmplusplus_small/large`) for the same embeddings via `transformers` with no clash.
- LightGBM / CARP are optional; missing backends simply don't appear in `plm-classifier list`.
- A CUDA GPU is used automatically when available.
