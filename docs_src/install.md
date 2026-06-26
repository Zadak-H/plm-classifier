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

## ESM-C setup (one time)

ESM-C (`esmc_300m`, `esmc_600m`) uses EvolutionaryScale's `esm` SDK which conflicts with
`fair-esm` and requires Python ≥ 3.10. **You do not need to switch environments manually** —
the tool auto-discovers any conda env that has the SDK and uses it via subprocess.

Set it up once:

```bash
conda create -n esmc python=3.10 -y
conda activate esmc
pip install esm httpx
```

After that, select `esmc_300m` or `esmc_600m` anywhere (GUI or CLI) and it works automatically.
You will see `[ESM-C] using: /path/to/esmc/bin/python3` in the log.

## Notes on other optional backends

- LightGBM / CARP are optional; missing backends simply don't appear in `plm-classifier list`.
- A CUDA GPU is used automatically when available.
