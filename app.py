#!/usr/bin/env python3
"""PLM-Classifier — Streamlit GUI.

Point-and-click classification: upload a CSV, pick the label + (optional) extra
columns, choose pLMs and classifiers, run a size-aware Optuna search, view the
report, and rank candidate sequences by class probability. Launch with::

    plm-classifier gui
    # or: streamlit run app.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from plm_classifier.config import RunConfig
from plm_classifier.core import COMPUTED_FEATURE_SOURCES, TABULAR_FEATURE_SOURCE
from plm_classifier.registry import DEFAULT_MODELS, available_models, available_plms

st.set_page_config(page_title="PLM-Classifier", layout="wide")


def _idx(options, value, default=0):
    try:
        return options.index(value)
    except (ValueError, AttributeError):
        return default


WORKSPACE = Path("plm_classifier_workspace")
WORKSPACE.mkdir(exist_ok=True)
ss = st.session_state
ss.setdefault("train_csv", None)
ss.setdefault("columns", [])

st.title("🧬 PLM-Classifier — sequence → class label")
st.caption("Binary (yes/no) or multiclass. Upload data, pick features + models, train, rank candidates. No coding required.")

tabs = st.tabs(["1. Data", "2. Features", "3. Models", "4. Search", "5. Run", "6. Results", "7. Predict"])

with tabs[0]:
    st.header("Training data")
    up = st.file_uploader("Upload a training CSV", type=["csv"])
    if up is not None:
        path = WORKSPACE / "train.csv"
        path.write_bytes(up.getbuffer())
        ss["train_csv"] = str(path)
    if ss.get("train_csv"):
        df = pd.read_csv(ss["train_csv"])
        ss["columns"] = list(df.columns)
        st.dataframe(df.head(20), use_container_width=True)
        cols = ss["columns"]
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        c1, c2, c3 = st.columns(3)
        ss["seq_col"] = c1.selectbox("Sequence column", cols, index=_idx(cols, ss.get("seq_col", "Protein_Seq")))
        ss["target_col"] = c2.selectbox("Label column", cols, index=_idx(cols, ss.get("target_col", cols[-1])))
        ss["id_col"] = c3.selectbox("ID column (optional)", ["<none>"] + cols, index=0)
        # positive class options from the chosen label column
        try:
            label_vals = sorted(df[ss["target_col"]].astype(str).dropna().unique().tolist())
        except Exception:
            label_vals = []
        c4, c5, c6 = st.columns(3)
        ss["positive_class"] = c4.selectbox(
            "Positive / ranking class", ["<auto: last>"] + label_vals, index=0,
            help="Candidates are ranked by this class's probability.",
        )
        ss["group_col"] = c5.selectbox("Group column (optional, leakage-safe CV)", ["<none>"] + cols, index=0)
        ss["extra_feature_cols"] = c6.multiselect("Extra numeric inputs (pH, temp, …)", numeric_cols)
        ss["categorical_cols"] = st.multiselect("Extra categorical inputs", [c for c in cols])
        if label_vals:
            st.caption(f"Detected {len(label_vals)} classes: {', '.join(label_vals[:10])}{' …' if len(label_vals) > 10 else ''}")
    else:
        st.info("Upload a CSV to begin. The label column can be yes/no or any set of classes.")

with tabs[1]:
    st.header("Feature sources")
    plm_choices = available_plms()
    encodings = sorted(COMPUTED_FEATURE_SOURCES)
    has_extra = bool(ss.get("extra_feature_cols") or ss.get("categorical_cols"))
    extra_opt = [TABULAR_FEATURE_SOURCE] if has_extra else []
    default_feats = [f for f in ["esm2"] if f in plm_choices] or plm_choices[:1]
    ss["feature_sources"] = st.multiselect(
        "Choose feature sources (pLMs / encodings / tabular)",
        plm_choices + encodings + extra_opt, default=ss.get("feature_sources", default_feats),
    )
    ss["embedding_dir"] = st.text_input("Embedding directory (npz banks)", ss.get("embedding_dir", "embeddings"))
    st.subheader("Embedding extraction")
    emb_dir = Path(ss.get("embedding_dir", "embeddings"))
    learned = [f for f in ss["feature_sources"] if f not in COMPUTED_FEATURE_SOURCES and f != TABULAR_FEATURE_SOURCE]
    for f in learned:
        present = (emb_dir / f"{f}.npz").exists()
        c = st.columns([3, 1])
        c[0].write(f"`{f}.npz` — {'✅ found' if present else '⚠️ missing'}")
        if not present and ss.get("train_csv") and ss.get("seq_col"):
            if c[1].button(f"Extract {f}", key=f"extract_{f}"):
                from plm_classifier.embeddings.extract import extract_from_csv

                with st.spinner(f"Extracting {f} (first run downloads the model)…"):
                    try:
                        _, n_new, n_cached = extract_from_csv(f, ss["train_csv"], ss["seq_col"], str(emb_dir / f"{f}.npz"))
                        st.success(f"{f}: {n_new} computed, {n_cached} cached")
                    except Exception as exc:
                        st.error(f"Extraction failed: {exc}")
    st.caption("onehot / blosum62 need aligned, equal-length sequences; tabular uses your extra columns.")

with tabs[2]:
    st.header("Classifiers")
    models = available_models()
    ss["models"] = st.multiselect(
        "Choose classifiers (classical + deep). Models too costly for the dataset size are auto-skipped.",
        models, default=[m for m in (ss.get("models") or DEFAULT_MODELS) if m in models],
    )
    st.caption("Deep models: `mlp_torch` (FNN on embeddings), `cnn1d` (1D-CNN on onehot/blosum).")

with tabs[3]:
    st.header("Search settings")
    c1, c2, c3 = st.columns(3)
    ss["metric"] = c1.selectbox("Primary metric",
                                ["roc_auc", "pr_auc", "f1", "f1_macro", "accuracy", "balanced_accuracy", "mcc", "log_loss"],
                                index=0)
    ss["auto_size"] = c2.checkbox("Auto-tune by dataset size", value=ss.get("auto_size", True))
    ss["standard_search"] = c3.checkbox("Standard (single sources only)", value=ss.get("standard_search", False))
    c4, c5, c6 = st.columns(3)
    override = c4.checkbox("Override trial budget", value=False)
    ss["n_trials"] = c4.number_input("Trials", 5, 1000, ss.get("n_trials") or 50) if override else None
    ss["top_ensemble"] = c5.number_input("Ensemble size (uncertainty)", 1, 20, ss.get("top_ensemble", 5))
    ss["no_uncertainty"] = c6.checkbox("Disable uncertainty", value=ss.get("no_uncertainty", False))
    ss["use_gpu"] = st.checkbox("Use GPU for boosting (xgb/lgb)", value=ss.get("use_gpu", False))

with tabs[4]:
    st.header("Run training")
    ss["out_dir"] = st.text_input("Output run directory", ss.get("out_dir", "runs/gui_run"))
    if st.button("🚀 Run training", type="primary"):
        if not ss.get("train_csv"):
            st.error("Upload a CSV first (tab 1).")
        else:
            cfg = RunConfig(
                csv=ss["train_csv"], seq_col=ss["seq_col"], target_col=ss["target_col"],
                id_col=None if ss.get("id_col", "<none>") == "<none>" else ss["id_col"],
                group_col=None if ss.get("group_col", "<none>") == "<none>" else ss["group_col"],
                positive_class=None if ss.get("positive_class", "<auto: last>") == "<auto: last>" else ss["positive_class"],
                extra_feature_cols=ss.get("extra_feature_cols", []), categorical_cols=ss.get("categorical_cols", []),
                feature_sources=ss["feature_sources"], embedding_dir=ss.get("embedding_dir", "embeddings"),
                models=ss["models"], metric=ss["metric"], auto_size=ss["auto_size"],
                standard_search=ss["standard_search"], n_trials=ss.get("n_trials"),
                top_ensemble=int(ss["top_ensemble"]), no_uncertainty=ss["no_uncertainty"],
                use_gpu=ss["use_gpu"], out_dir=ss["out_dir"],
            )
            try:
                cfg.validate()
            except Exception as exc:
                st.error(f"Invalid config: {exc}"); st.stop()
            cfg_path = WORKSPACE / "run_config.yaml"; cfg.to_yaml(cfg_path)
            log_path = WORKSPACE / "train.log"
            st.info("Training started. Live log below.")
            log_box = st.empty()
            with open(log_path, "w") as logf:
                proc = subprocess.Popen([sys.executable, "-m", "plm_classifier.cli", "train", str(cfg_path)],
                                        stdout=logf, stderr=subprocess.STDOUT)
                import time
                while proc.poll() is None:
                    log_box.code(Path(log_path).read_text()[-4000:]); time.sleep(1.5)
                log_box.code(Path(log_path).read_text()[-4000:])
            if proc.returncode == 0:
                st.success(f"Done. Open tab 6 (Results) at {ss['out_dir']}.")
            else:
                st.error("Training failed; see log above.")

with tabs[5]:
    st.header("Results")
    run_dir = Path(st.text_input("Run directory", ss.get("out_dir", "runs/gui_run")))
    if (run_dir / "run_report.json").exists():
        report = json.loads((run_dir / "run_report.json").read_text())
        st.subheader(f"{report.get('run_name')} — best model: {report.get('best_model_name')}")
        st.write(f"Classes: **{', '.join(report.get('classes') or [])}** · ranked by P(**{report.get('positive_class')}**) "
                 f"· tier: {report.get('size_tier')} · CV: {report.get('cv_strategy')}")
        st.table(pd.DataFrame([report.get("oof_metrics", {})]).T.rename(columns={0: "value"}))
        for img in report.get("plots", []):
            p = run_dir / "plots" / Path(img).name
            if p.exists():
                st.image(str(p))
        for fname in ["best_model.joblib", "oof_predictions.csv", "candidate_predictions.csv"]:
            fp = run_dir / fname
            if fp.exists():
                st.download_button(f"Download {fname}", fp.read_bytes(), file_name=fname)
    else:
        st.info("No run_report.json found in that directory yet.")

with tabs[6]:
    st.header("Rank candidate sequences")
    pred_run_dir = st.text_input("Saved run directory", ss.get("out_dir", "runs/gui_run"), key="pred_run")
    cand = st.file_uploader("Upload candidate CSV", type=["csv"], key="cand")
    pseq = st.text_input("Candidate sequence column", "Sequence")
    cand_emb = st.text_input("Candidate embedding directory (optional)", "")
    pos = st.text_input("Rank by class (optional, overrides run default)", "")
    top_n = st.number_input("Top N", 1, 100000, 100)
    if st.button("Rank candidates"):
        if cand is None:
            st.error("Upload a candidate CSV first.")
        else:
            cand_path = WORKSPACE / "candidates.csv"; cand_path.write_bytes(cand.getbuffer())
            from plm_classifier.predict import score_candidates_from_run

            with st.spinner("Scoring candidates…"):
                try:
                    out = score_candidates_from_run(
                        run_dir=pred_run_dir, candidate_csv=str(cand_path), predict_seq_col=pseq,
                        candidate_embedding_dir=cand_emb or None, top_n=int(top_n), positive_class=pos or None,
                    )
                    ranked = pd.read_csv(out)
                    st.success(f"Ranked {len(ranked)} candidates.")
                    st.dataframe(ranked.head(50), use_container_width=True)
                    st.download_button("Download candidate_predictions.csv", out.read_bytes(),
                                       file_name="candidate_predictions.csv")
                except Exception as exc:
                    st.error(f"Ranking failed: {exc}")
