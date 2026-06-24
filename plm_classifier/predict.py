#!/usr/bin/env python3
"""Score / rank candidate sequences with a saved classifier run (probability-based)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd

from .core import (
    TABULAR_FEATURE_SOURCE,
    assemble_feature_matrices,
    load_embedding_banks,
    load_json,
    proba_score_dataframe,
    transform_feature_mode,
)


def write_top_tables(df: pd.DataFrame, out_dir: Path, top_n: int) -> None:
    valid = df[df["rank"].notna()].copy()
    sizes: list[int] = []
    for size in [10, 50, 100, top_n]:
        if size not in sizes:
            sizes.append(size)
    for size in sizes:
        valid.head(size).to_csv(out_dir / f"top_{size}.csv", index=False)


def score_candidates_from_run(
    run_dir: str | Path,
    candidate_csv: str | Path,
    predict_seq_col: Optional[str] = None,
    candidate_embedding_dir: Optional[str] = None,
    top_n: int = 100,
    out_dir: Optional[str | Path] = None,
    positive_class: Optional[str] = None,
) -> Path:
    run_dir = Path(run_dir)
    summary = load_json(run_dir / "run_summary.json")
    best_model = joblib.load(run_dir / "best_model.joblib")
    uncertainty_enabled = bool(summary.get("uncertainty_enabled", True))

    ensemble = None
    ens_path = run_dir / "uncertainty_ensemble.joblib"
    if uncertainty_enabled and ens_path.exists():
        ensemble = joblib.load(ens_path)

    tabular_encoder = None
    tab_path = run_dir / "tabular_encoder.joblib"
    if summary.get("has_tabular") and tab_path.exists():
        tabular_encoder = joblib.load(tab_path)

    candidate_df = pd.read_csv(candidate_csv)
    predict_seq_col = predict_seq_col or summary.get("predict_seq_col") or summary["train_seq_col"]
    if predict_seq_col not in candidate_df.columns:
        raise ValueError(f"Prediction sequence column '{predict_seq_col}' not found in {candidate_csv}")

    feature_sources = summary.get("deployment_feature_sources") or summary["feature_sources"]
    embedding_dir = candidate_embedding_dir or summary.get("predict_embedding_dir") or summary.get("embedding_dir")
    banks = load_embedding_banks(feature_sources=feature_sources, embedding_dir=embedding_dir)

    tabular_matrix = None
    if TABULAR_FEATURE_SOURCE in [s.lower() for s in feature_sources]:
        if tabular_encoder is None:
            raise ValueError("Run uses tabular features but no tabular_encoder.joblib was found")
        tabular_matrix = tabular_encoder.transform(candidate_df)

    X_raw, missing_any, _, _ = assemble_feature_matrices(
        df=candidate_df, seq_col=predict_seq_col, feature_sources=feature_sources,
        embedding_banks=banks, expected_sequence_length=summary.get("expected_sequence_length"),
        tabular_matrix=tabular_matrix,
    )

    feature_modes = {best_model.feature_mode}
    if ensemble is not None:
        feature_modes |= {m.feature_mode for m in ensemble.fitted_models}
    X_by_mode: Dict[str, Dict[str, np.ndarray]] = {
        mode: transform_feature_mode(X_raw, None, mode, best_model.wt_by_source or None)
        for mode in sorted(feature_modes)
    }

    classes = np.array(best_model.classes)
    n_classes = best_model.n_classes
    n = len(candidate_df)
    proba = np.full((n, n_classes), np.nan, dtype=float)
    ens_std = np.full(n, np.nan, dtype=float)
    valid = ~missing_any
    if valid.any():
        proba[valid] = best_model.predict_proba({k: v[valid] for k, v in X_by_mode[best_model.feature_mode].items()})
        if uncertainty_enabled and ensemble is not None:
            mean_v, std_v = ensemble.predict_proba(
                {mode: {k: v[valid] for k, v in smap.items()} for mode, smap in X_by_mode.items()}
            )
            proba[valid] = mean_v
            ens_std[valid] = std_v

    seen = candidate_df[predict_seq_col].astype(str).str.strip().isin(set(best_model.train_sequences)).to_numpy()
    ranked = proba_score_dataframe(
        df=candidate_df, run_name=summary["run_name"], proba=proba, classes=classes,
        positive_class=positive_class or summary.get("positive_class"),
        ensemble_std=ens_std, seen_in_train=seen, missing_any_feature=missing_any, rank=True,
    )

    out_dir = Path(out_dir) if out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "candidate_predictions.csv"
    ranked.to_csv(out_path, index=False)
    write_top_tables(ranked, out_dir=out_dir, top_n=top_n)
    print(f"Saved ranked candidates to: {out_path} (ranked by P({ranked['ranked_class'].iloc[0]}))")
    return out_path
