#!/usr/bin/env python3
"""Training orchestrator for PLM-Classifier: RunConfig -> run directory + report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler

from .config import RunConfig
from .core import (
    TABULAR_FEATURE_SOURCE,
    FittedEnsemble,
    FittedRunModel,
    all_nonempty_feature_subsets,
    assemble_feature_matrices,
    cross_val_predict_proba,
    json_default,
    load_embedding_banks,
    make_cv_splits,
    normalize_feature_source,
    prepare_supervised_dataframe,
    print_header,
    proba_score_dataframe,
    save_json,
    seed_everything,
    transform_feature_mode,
)
from .features import build_tabular_encoder
from .metrics import metric_direction, sort_descending_for_metric
from .registry import MODEL_REGISTRY, eligible_models
from .search import SearchContext, build_fixed_trial_estimator, make_objective
from .sizing import SizeProfile, profile_for_n


def _resolve_profile(cfg: RunConfig, n: int) -> SizeProfile:
    profile = profile_for_n(n)
    if not cfg.auto_size:
        profile.cv_strategy = cfg.cv_strategy
        profile.cv_splits = cfg.cv_splits
    if cfg.n_trials is not None:
        profile.trial_budget = int(cfg.n_trials)
    return profile


def run_training(cfg: RunConfig) -> Path:
    cfg.validate()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.random_state)

    feature_sources = [normalize_feature_source(s) for s in cfg.feature_sources]
    has_tabular = TABULAR_FEATURE_SOURCE in feature_sources

    print_header("Loading supervised CSV")
    raw_df = pd.read_csv(cfg.csv)
    if cfg.seq_col not in raw_df.columns:
        raise ValueError(f"Sequence column '{cfg.seq_col}' not found in {cfg.csv}")
    if cfg.target_col not in raw_df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found in {cfg.csv}")

    prepared_df = prepare_supervised_dataframe(
        df=raw_df, seq_col=cfg.seq_col, target_col=cfg.target_col,
        replicate_policy=cfg.replicate_policy, id_col=cfg.id_col,
    )

    # encode labels -> 0..K-1 (sorted for determinism)
    label_str = prepared_df[cfg.target_col].astype(str).to_numpy()
    classes = np.array(sorted(np.unique(label_str)))
    n_classes = len(classes)
    if n_classes < 2:
        raise RuntimeError(f"Need >= 2 classes, found {n_classes}: {classes.tolist()}")
    cls_index = {c: i for i, c in enumerate(classes)}
    y = np.array([cls_index[c] for c in label_str], dtype=int)

    if cfg.group_col and cfg.group_col in prepared_df.columns:
        groups = prepared_df[cfg.group_col].astype(str).to_numpy()
        group_col_used = cfg.group_col
    else:
        groups = prepared_df[cfg.seq_col].astype(str).to_numpy()
        group_col_used = cfg.seq_col

    print(f"Rows raw: {len(raw_df)} | after replicate '{cfg.replicate_policy}': {len(prepared_df)}")
    print(f"Classes ({n_classes}): {classes.tolist()}  | counts: {np.bincount(y).tolist()}")

    tabular_encoder = None
    if has_tabular:
        tabular_encoder = build_tabular_encoder(prepared_df, cfg.extra_feature_cols, cfg.categorical_cols)
        if tabular_encoder is None:
            raise ValueError("tabular feature source requested but no extra/categorical columns provided")

    print_header("Loading feature banks")
    profile = _resolve_profile(cfg, len(prepared_df))
    print(f"Dataset size profile: {profile.describe()}")
    embedding_banks = load_embedding_banks(feature_sources, cfg.embedding_dir, None, mmap=profile.mmap_embeddings)
    for name, bank in embedding_banks.items():
        print(f"- {name}: dim={bank.dim} | {bank.path}")

    tabular_matrix = tabular_encoder.transform(prepared_df) if tabular_encoder is not None else None
    X_by_source_raw, missing_any, source_missing_counts, expected_len = assemble_feature_matrices(
        df=prepared_df, seq_col=cfg.seq_col, feature_sources=feature_sources,
        embedding_banks=embedding_banks, tabular_matrix=tabular_matrix,
    )
    rows_before = len(prepared_df)
    if missing_any.any():
        keep = ~missing_any
        print(f"Dropping {int(missing_any.sum())} rows with missing learned features")
        prepared_df = prepared_df.loc[keep].reset_index(drop=True)
        y = y[keep]; groups = groups[keep]; label_str = label_str[keep]
        for s in list(X_by_source_raw.keys()):
            X_by_source_raw[s] = X_by_source_raw[s][keep]
    if len(prepared_df) < 4:
        raise RuntimeError("Not enough rows remain after feature coverage filtering")

    # WT for delta modes
    wt_index = None
    wt_by_source = None
    if cfg.wt_sequence is not None:
        seqs = prepared_df[cfg.seq_col].astype(str).tolist()
        wt = str(cfg.wt_sequence).strip()
        if wt not in seqs:
            raise ValueError("WT sequence not found in filtered training data")
        wt_index = seqs.index(wt)
        wt_by_source = {s: m[wt_index].copy() for s, m in X_by_source_raw.items()}

    if cfg.standard_search:
        feature_modes = ["raw"]
    elif cfg.feature_mode_options:
        feature_modes = list(dict.fromkeys(cfg.feature_mode_options))
    elif wt_index is not None:
        feature_modes = ["raw", "delta", "raw_plus_delta"]
    else:
        feature_modes = ["raw"]
    if any(m != "raw" for m in feature_modes) and wt_index is None:
        raise ValueError("Delta-based feature modes require wt_sequence")

    X_by_mode = {m: transform_feature_mode(X_by_source_raw, wt_index, m, wt_by_source) for m in feature_modes}
    feature_subsets = ([(s,) for s in feature_sources] if cfg.standard_search
                       else all_nonempty_feature_subsets(feature_sources))

    n_rows = len(prepared_df)
    elig = eligible_models(cfg.models, n_rows)
    dropped = [m for m in cfg.models if m not in elig and m in MODEL_REGISTRY]
    if dropped:
        print(f"Models excluded for n={n_rows}: {', '.join(dropped)}")
    if not elig:
        raise RuntimeError("No eligible models remain; relax the model list")

    rng = np.random.RandomState(cfg.random_state)
    if profile.tune_subsample and n_rows > profile.tune_subsample:
        cand = np.sort(rng.choice(n_rows, size=profile.tune_subsample, replace=False))
        tune_idx = cand if len(np.unique(y[cand])) == n_classes else np.arange(n_rows)
        print(f"Tuning on {len(tune_idx)} / {n_rows} rows; deploying on all rows")
    else:
        tune_idx = np.arange(n_rows)

    X_by_mode_tune = {m: {s: X_by_mode[m][s][tune_idx] for s in X_by_mode[m]} for m in feature_modes}
    y_tune = y[tune_idx]; groups_tune = groups[tune_idx]
    splits, uses_group_cv = make_cv_splits(
        n_samples=len(tune_idx), y=y_tune, groups=groups_tune, cv_splits=profile.cv_splits,
        random_state=cfg.random_state, strategy=profile.cv_strategy,
        n_repeats=profile.n_repeats, holdout_fraction=profile.holdout_fraction,
    )

    print_header("Search space summary")
    print(f"Feature sources: {', '.join(feature_sources)} | subsets: {len(feature_subsets)} | modes: {', '.join(feature_modes)}")
    print(f"Eligible models: {', '.join(elig)}")
    print(f"Metric: {cfg.metric} | CV: {profile.cv_strategy} ({len(splits)} folds) | trials: {profile.trial_budget}")

    ctx = SearchContext(
        X_by_mode=X_by_mode_tune, y=y_tune, classes=classes, n_classes=n_classes, groups=groups_tune,
        splits=splits, feature_subsets=feature_subsets, feature_modes=feature_modes, eligible_models=elig,
        metric_name=cfg.metric, random_state=cfg.random_state, use_gpu=cfg.use_gpu, standard_search=cfg.standard_search,
    )

    print_header("Running Optuna search")
    sampler = TPESampler(seed=cfg.random_state, multivariate=True)
    study = optuna.create_study(direction=metric_direction(cfg.metric), sampler=sampler)
    study.optimize(make_objective(ctx), n_trials=profile.trial_budget, timeout=cfg.timeout, show_progress_bar=True)

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise RuntimeError("No completed Optuna trials (all pruned/failed)")
    descending = sort_descending_for_metric(cfg.metric)
    complete = sorted(complete, key=lambda t: float(t.value), reverse=descending)
    best = complete[0]
    best_subset = tuple(best.user_attrs["feature_subset"])
    best_mode = str(best.user_attrs["feature_mode"])
    best_X_full = np.concatenate([X_by_mode[best_mode][s] for s in best_subset], axis=1)
    best_X_tune = np.concatenate([X_by_mode_tune[best_mode][s] for s in best_subset], axis=1)

    best_estimator, best_meta = build_fixed_trial_estimator(
        frozen_trial=best, eligible_models=elig, X=best_X_tune, n_splits=len(splits),
        random_state=cfg.random_state + best.number, use_gpu=cfg.use_gpu, n_classes=n_classes,
        standard_search=cfg.standard_search,
    )

    print_header("Evaluating best model (OOF on tuning set)")
    best_metrics, best_oof_proba, best_fold_ids, best_fold_rows = cross_val_predict_proba(
        best_estimator, best_X_tune, y_tune, splits, n_classes, classes
    )

    # search history
    hist = []
    for t in complete:
        hist.append({
            "trial_number": t.number, "value": t.value,
            "feature_subset": "+".join(t.user_attrs.get("feature_subset", [])),
            "feature_mode": t.user_attrs.get("feature_mode"),
            "model_name": t.user_attrs.get("meta", {}).get("model_name"),
            **{f"metric__{k}": v for k, v in t.user_attrs.get("metrics", {}).items()},
            "meta_json": json.dumps(t.user_attrs.get("meta", {}), default=json_default),
        })
    pd.DataFrame(hist).sort_values("value", ascending=not descending).to_csv(out_dir / "search_history.csv", index=False)

    print_header("Fitting best model (full data) + ensemble")
    best_estimator.fit(best_X_full, y)
    classes_tuple = tuple(str(c) for c in classes)
    best_artifact = FittedRunModel(
        run_name=out_dir.name, trial_number=best.number, score=float(best.value), metric_name=cfg.metric,
        model_name=best_meta["model_name"], feature_sources=best_subset, feature_mode=best_mode,
        estimator=best_estimator, classes=classes_tuple, n_classes=n_classes,
        train_sequences=tuple(prepared_df[cfg.seq_col].astype(str).tolist()),
        expected_sequence_length=expected_len, wt_by_source={} if wt_by_source is None else wt_by_source,
    )
    joblib.dump(best_artifact, out_dir / "best_model.joblib")
    if tabular_encoder is not None:
        joblib.dump(tabular_encoder, out_dir / "tabular_encoder.joblib")

    ensemble = None
    ensemble_members: List[FittedRunModel] = []
    ensemble_oof_std = np.full(len(best_oof_proba), np.nan, dtype=float)
    if not cfg.no_uncertainty:
        oof_probas = []
        rows = []
        for t in complete[: max(1, min(cfg.top_ensemble, len(complete)))]:
            fsub = tuple(t.user_attrs["feature_subset"]); fmode = str(t.user_attrs["feature_mode"])
            Xt_tune = np.concatenate([X_by_mode_tune[fmode][s] for s in fsub], axis=1)
            Xt_full = np.concatenate([X_by_mode[fmode][s] for s in fsub], axis=1)
            est_t, meta_t = build_fixed_trial_estimator(
                frozen_trial=t, eligible_models=elig, X=Xt_tune, n_splits=len(splits),
                random_state=cfg.random_state + t.number, use_gpu=cfg.use_gpu, n_classes=n_classes,
                standard_search=cfg.standard_search,
            )
            _, oof_t, _, _ = cross_val_predict_proba(est_t, Xt_tune, y_tune, splits, n_classes, classes)
            oof_probas.append(oof_t)
            est_t.fit(Xt_full, y)
            ensemble_members.append(FittedRunModel(
                run_name=out_dir.name, trial_number=t.number, score=float(t.value), metric_name=cfg.metric,
                model_name=meta_t["model_name"], feature_sources=fsub, feature_mode=fmode, estimator=est_t,
                classes=classes_tuple, n_classes=n_classes,
                train_sequences=tuple(prepared_df[cfg.seq_col].astype(str).tolist()),
                expected_sequence_length=expected_len, wt_by_source={} if wt_by_source is None else wt_by_source,
            ))
            rows.append({"trial_number": t.number, "score": float(t.value), "feature_subset": "+".join(fsub),
                         "feature_mode": fmode, "model_name": meta_t["model_name"]})
        ensemble = FittedEnsemble(run_name=out_dir.name, fitted_models=ensemble_members, n_classes=n_classes)
        joblib.dump(ensemble, out_dir / "uncertainty_ensemble.joblib")
        pd.DataFrame(rows).to_csv(out_dir / "top_ensemble_members.csv", index=False)
        stacked = np.stack(oof_probas, axis=0)             # (members, n_tune, K)
        ensemble_oof_std = stacked.std(axis=0, ddof=0).mean(axis=1)
        deployment_sources = sorted({s for m in ensemble_members for s in m.feature_sources})
    else:
        deployment_sources = sorted(best_subset)

    # OOF table
    oof_df = proba_score_dataframe(
        df=prepared_df.iloc[tune_idx].reset_index(drop=True), run_name=out_dir.name,
        proba=best_oof_proba, classes=classes, positive_class=cfg.positive_class,
        fold_ids=best_fold_ids, true_labels=classes[y_tune], ensemble_std=ensemble_oof_std,
    )
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(best_fold_rows).to_csv(out_dir / "fold_metrics.csv", index=False)

    # train predictions (full)
    train_proba = best_artifact.predict_proba(X_by_mode[best_mode])
    train_df = proba_score_dataframe(
        df=prepared_df.copy(), run_name=out_dir.name, proba=train_proba, classes=classes,
        positive_class=cfg.positive_class, true_labels=classes[y],
        seen_in_train=np.ones(len(prepared_df), dtype=bool),
    )
    train_df.to_csv(out_dir / "train_predictions.csv", index=False)

    save_json(out_dir / "coverage_report.json", {
        "feature_sources": feature_sources, "source_missing_counts_before_drop": source_missing_counts,
        "rows_before_feature_drop": rows_before, "rows_after_feature_drop": int(len(prepared_df)),
        "expected_sequence_length": int(expected_len),
    })

    run_summary = {
        "task": "classification", "run_name": out_dir.name, "csv": cfg.csv, "predict_csv": cfg.predict_csv,
        "train_seq_col": cfg.seq_col, "predict_seq_col": cfg.predict_seq_col or cfg.seq_col,
        "target_col": cfg.target_col, "id_col": cfg.id_col, "group_col": group_col_used,
        "classes": classes.tolist(), "n_classes": int(n_classes), "class_counts": np.bincount(y).tolist(),
        "positive_class": cfg.positive_class or str(classes[-1]),
        "replicate_policy": cfg.replicate_policy, "feature_sources": feature_sources,
        "deployment_feature_sources": deployment_sources, "extra_feature_cols": cfg.extra_feature_cols,
        "categorical_cols": cfg.categorical_cols, "has_tabular": has_tabular,
        "feature_modes_searched": feature_modes, "standard_search": bool(cfg.standard_search),
        "uncertainty_enabled": bool(not cfg.no_uncertainty), "size_tier": profile.tier,
        "cv_strategy": profile.cv_strategy, "trial_budget": profile.trial_budget, "tuned_on_rows": int(len(tune_idx)),
        "metric": cfg.metric, "metric_direction": metric_direction(cfg.metric), "eligible_models": elig,
        "models_requested": cfg.models, "models_dropped": dropped, "best_trial_number": int(best.number),
        "best_trial_value": float(best.value), "best_feature_subset": list(best_subset),
        "best_feature_mode": best_mode, "best_model_name": best_meta["model_name"], "best_params": best.params,
        "best_meta": best_meta, "best_oof_metrics": best_metrics, "top_ensemble_size": int(len(ensemble_members)),
        "n_raw_rows": int(len(raw_df)), "n_training_rows": int(len(prepared_df)),
        "embedding_dir": cfg.embedding_dir, "predict_embedding_dir": cfg.predict_embedding_dir or cfg.embedding_dir,
        "expected_sequence_length": int(expected_len), "wt_sequence": cfg.wt_sequence,
        "uses_group_cv": bool(uses_group_cv), "coverage_report_file": "coverage_report.json",
    }
    save_json(out_dir / "run_summary.json", run_summary)
    cfg.to_yaml(out_dir / "run_config.yaml")

    if cfg.predict_csv:
        from .predict import score_candidates_from_run

        print_header(f"Scoring candidates: {cfg.predict_csv}")
        score_candidates_from_run(
            run_dir=out_dir, candidate_csv=cfg.predict_csv, predict_seq_col=cfg.predict_seq_col,
            candidate_embedding_dir=cfg.predict_embedding_dir, top_n=100, out_dir=out_dir,
        )

    try:
        from .report import build_report

        build_report(out_dir, cfg.metric)
    except Exception as exc:  # pragma: no cover
        print(f"[warn] report generation skipped: {exc}")

    print_header("Done")
    print(f"Best {cfg.metric}: {best.value:.4f} | model: {best_meta['model_name']} | "
          f"features: {'+'.join(best_subset)} ({best_mode}) | classes: {classes.tolist()}")
    print(f"Outputs: {out_dir}")
    return out_dir


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train a classifier from a RunConfig YAML")
    parser.add_argument("config")
    args = parser.parse_args(argv)
    run_training(RunConfig.from_yaml(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
