#!/usr/bin/env python3
"""Core utilities for PLM-Classifier.

Feature/embedding machinery is identical to PLM-Regressor; the modelling parts are
classification-specific: stratified CV, out-of-fold *probabilities*, label-aware
replicate aggregation, and probability-returning fitted artifacts.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)

import warnings

from .metrics import compute_metrics, predictive_entropy

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_SGKF = True
except Exception:  # pragma: no cover
    StratifiedGroupKFold = None
    HAS_SGKF = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


COMPUTED_FEATURE_SOURCES = {"onehot", "blosum62"}
TABULAR_FEATURE_SOURCE = "tabular"
AA_ALPHABET: Tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AA_ALPHABET)}

_BLOSUM62_ROWS = {
    "A": [4, 0, -2, -1, -2, 0, -2, -1, -1, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
    "C": [0, 9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    "D": [-2, -3, 6, 2, -3, -1, -1, -3, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
    "E": [-1, -4, 2, 5, -3, -2, 0, -3, 1, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
    "F": [-2, -2, -3, -3, 6, -3, -3, -1, -3, 0, 0, -3, 0, 1, -3, -2, -2, 1, 3, -1],
    "G": [0, -3, -1, -2, -3, 6, -2, -4, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
    "H": [-2, -3, -1, 0, -3, -2, 8, -3, -1, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
    "I": [-1, -1, -3, -3, -1, -4, -3, 4, -3, 2, 1, -3, 1, 0, -3, -2, -1, -3, -1, 3],
    "K": [-1, -3, -1, 1, -3, -2, -1, -3, 5, -2, -3, 1, -1, -3, -1, 0, -1, -3, -2, -2],
    "L": [-1, -1, -3, -3, 0, -4, -3, 2, -2, 4, 2, -2, 2, 0, -3, -2, -1, -2, -1, 1],
    "M": [-1, -1, -4, -3, 0, -4, -3, 1, -3, 2, 5, -2, 3, 0, -2, -1, -1, -1, -1, 1],
    "N": [-1, -3, -1, 1, -3, -2, -1, -3, 1, -2, -2, 6, -2, -4, -2, 0, -1, -4, -2, -3],
    "P": [-1, -1, -3, -2, 0, -3, -2, 1, -1, 2, 3, -2, 7, -1, -2, -1, -1, -1, -1, 1],
    "Q": [-2, -2, -3, -3, 1, -3, -1, 0, -3, 0, 0, -4, -1, 5, -1, -2, -2, 1, 3, -1],
    "R": [-1, -3, -1, -1, -3, -2, -2, -3, -1, -3, -2, -2, -2, -1, 5, -1, -1, -3, -2, -3],
    "S": [1, -1, 0, 0, -2, 0, -1, -2, 0, -2, -1, 0, -1, -2, -1, 4, 1, -3, -2, -2],
    "T": [0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
    "W": [-3, -2, -4, -3, 1, -2, -2, -3, -3, -2, -1, -4, -1, 1, -3, -3, -2, 11, 2, -3],
    "Y": [-2, -2, -3, -2, 3, -3, 2, -1, -2, -1, -1, -2, -1, 3, -2, -2, -2, 2, 7, -1],
    "V": [0, -1, -3, -2, -1, -3, -3, 3, -2, 1, 1, -3, 1, -1, -3, -2, 0, -3, -1, 4],
}
BLOSUM62_MATRIX = np.asarray([_BLOSUM62_ROWS[aa] for aa in AA_ALPHABET], dtype=np.float32)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (Path,)):
        return str(obj)
    return str(obj)


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=json_default)


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def print_header(title: str) -> None:
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)


@dataclass
class EmbeddingBank:
    name: str
    path: str
    seq_to_vec: Dict[str, np.ndarray]
    dim: int


class IdentityTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class ColumnSelectorKBest(TransformerMixin, BaseEstimator):
    """KBest with classification score functions (f_classif / mutual_info_classif)."""

    def __init__(self, score_func: str = "f_classif", k: int = 100):
        self.score_func = score_func
        self.k = int(k)
        self.selector_: Optional[SelectKBest] = None

    def fit(self, X, y):
        k_eff = min(self.k, X.shape[1])
        func = f_classif if self.score_func == "f_classif" else mutual_info_classif
        self.selector_ = SelectKBest(score_func=func, k=k_eff)
        self.selector_.fit(X, y)
        return self

    def transform(self, X):
        if self.selector_ is None:
            raise RuntimeError("ColumnSelectorKBest not fitted")
        return self.selector_.transform(X)


def normalize_feature_source(name: str) -> str:
    lowered = name.strip().lower()
    if lowered in COMPUTED_FEATURE_SOURCES or lowered == TABULAR_FEATURE_SOURCE:
        return lowered
    return name.strip()


def load_embedding_npz(npz_path: str, mmap: bool = False) -> Tuple[List[str], np.ndarray]:
    data = np.load(npz_path, allow_pickle=True, mmap_mode="r" if mmap else None)
    keys = set(data.files)

    def to_list(values: Any) -> List[str]:
        if isinstance(values, np.ndarray):
            values = values.tolist()
        return [str(item).strip() for item in values]

    if "sequences" in keys and "embeddings" in keys:
        sequences = to_list(data["sequences"])
        embeddings = np.asarray(data["embeddings"])
    elif "seqs" in keys and "embeddings" in keys:
        sequences = to_list(data["seqs"])
        embeddings = np.asarray(data["embeddings"])
    elif "arr_0" in keys:
        obj = data["arr_0"]
        try:
            obj = obj.item()
        except Exception:
            pass
        if isinstance(obj, dict) and "embeddings" in obj and "sequences" in obj:
            sequences = to_list(obj["sequences"])
            embeddings = np.asarray(obj["embeddings"])
        elif isinstance(obj, dict):
            sequences = to_list(obj.keys())
            embeddings = np.stack([obj[seq] for seq in sequences], axis=0)
        else:
            raise ValueError(f"Unsupported arr_0 layout in {npz_path}")
    elif "seq_to_index" in keys and "embeddings" in keys:
        raw = data["seq_to_index"]
        if hasattr(raw, "item"):
            try:
                raw = raw.item()
            except Exception:
                pass
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, dict):
            raise ValueError(f"Unsupported seq_to_index type in {npz_path}")
        sequences = [str(key).strip() for key, _ in sorted(raw.items(), key=lambda item: int(item[1]))]
        embeddings = np.asarray(data["embeddings"])
    else:
        raise ValueError(f"Unsupported embedding format in {npz_path}. Keys={sorted(keys)}")

    if embeddings.ndim == 3:
        embeddings = embeddings.mean(axis=1)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings after pooling, got {embeddings.shape} in {npz_path}")
    return sequences, embeddings.astype(np.float32, copy=False)


def build_embedding_bank(npz_path: str, name: Optional[str] = None, mmap: bool = False) -> EmbeddingBank:
    sequences, embeddings = load_embedding_npz(npz_path, mmap=mmap)
    seq_to_vec = {str(seq).strip(): embeddings[idx] for idx, seq in enumerate(sequences)}
    return EmbeddingBank(name=name or Path(npz_path).stem, path=str(npz_path), seq_to_vec=seq_to_vec, dim=int(embeddings.shape[1]))


def resolve_embedding_paths(
    feature_sources: Sequence[str],
    embedding_dir: Optional[str] = None,
    explicit_embedding_paths: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    learned_sources = [
        normalize_feature_source(s)
        for s in feature_sources
        if normalize_feature_source(s) not in COMPUTED_FEATURE_SOURCES and normalize_feature_source(s) != TABULAR_FEATURE_SOURCE
    ]
    paths: Dict[str, str] = {}
    if explicit_embedding_paths:
        for raw_path in explicit_embedding_paths:
            path = Path(raw_path)
            paths[path.stem] = str(path)
    if embedding_dir:
        embedding_dir_path = Path(embedding_dir)
        for source in learned_sources:
            if source in paths:
                continue
            candidate = embedding_dir_path / f"{source}.npz"
            if not candidate.exists():
                raise FileNotFoundError(f"Expected embedding file for feature source '{source}' at {candidate}")
            paths[source] = str(candidate)
    missing = [s for s in learned_sources if s not in paths]
    if missing:
        raise ValueError("Missing embedding paths for learned feature sources: " + ", ".join(sorted(missing)))
    return paths


def load_embedding_banks(
    feature_sources: Sequence[str],
    embedding_dir: Optional[str] = None,
    explicit_embedding_paths: Optional[Sequence[str]] = None,
    mmap: bool = False,
) -> Dict[str, EmbeddingBank]:
    resolved = resolve_embedding_paths(feature_sources, embedding_dir, explicit_embedding_paths)
    return {name: build_embedding_bank(path, name=name, mmap=mmap) for name, path in resolved.items()}


def infer_sequence_length(sequences: Sequence[str], expected_length: Optional[int] = None) -> int:
    lengths = {len(str(s).strip()) for s in sequences}
    if not lengths:
        raise ValueError("Cannot infer sequence length from an empty collection")
    if len(lengths) != 1:
        raise ValueError(
            "One-hot and BLOSUM62 encodings require aligned fixed-length sequences. "
            f"Observed lengths: {sorted(lengths)}"
        )
    length = next(iter(lengths))
    if expected_length is not None and length != expected_length:
        raise ValueError(f"Sequence length mismatch: expected {expected_length}, observed {length}")
    return length


def encode_onehot_sequences(sequences: Sequence[str], expected_length: Optional[int] = None) -> np.ndarray:
    sequences = [str(s).strip() for s in sequences]
    seq_len = infer_sequence_length(sequences, expected_length=expected_length)
    encoded = np.zeros((len(sequences), seq_len, len(AA_ALPHABET)), dtype=np.float32)
    for r, seq in enumerate(sequences):
        for p, aa in enumerate(seq):
            j = AA_TO_INDEX.get(aa)
            if j is not None:
                encoded[r, p, j] = 1.0
    return encoded.reshape(len(sequences), seq_len * len(AA_ALPHABET))


def encode_blosum62_sequences(sequences: Sequence[str], expected_length: Optional[int] = None) -> np.ndarray:
    sequences = [str(s).strip() for s in sequences]
    seq_len = infer_sequence_length(sequences, expected_length=expected_length)
    encoded = np.zeros((len(sequences), seq_len, len(AA_ALPHABET)), dtype=np.float32)
    for r, seq in enumerate(sequences):
        for p, aa in enumerate(seq):
            j = AA_TO_INDEX.get(aa)
            if j is not None:
                encoded[r, p, :] = BLOSUM62_MATRIX[j]
    return encoded.reshape(len(sequences), seq_len * len(AA_ALPHABET))


def assemble_single_embedding(sequences: Sequence[str], bank: EmbeddingBank) -> Tuple[np.ndarray, np.ndarray]:
    clean = [str(s).strip() for s in sequences]
    matrix = np.zeros((len(clean), bank.dim), dtype=np.float32)
    missing = np.zeros(len(clean), dtype=bool)
    for i, s in enumerate(clean):
        vec = bank.seq_to_vec.get(s)
        if vec is None:
            missing[i] = True
        else:
            matrix[i] = vec
    return matrix, missing


def assemble_feature_matrices(
    df: pd.DataFrame,
    seq_col: str,
    feature_sources: Sequence[str],
    embedding_banks: Optional[Dict[str, EmbeddingBank]] = None,
    expected_sequence_length: Optional[int] = None,
    tabular_matrix: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, int], int]:
    sequences = df[seq_col].astype(str).str.strip().tolist()
    matrices: Dict[str, np.ndarray] = {}
    missing_any = np.zeros(len(df), dtype=bool)
    source_missing_counts: Dict[str, int] = {}
    computed_len: Optional[int] = None

    for raw_source in feature_sources:
        source = normalize_feature_source(raw_source)
        if source == "onehot":
            matrices[source] = encode_onehot_sequences(sequences, expected_length=expected_sequence_length)
            source_missing_counts[source] = 0
            computed_len = matrices[source].shape[1] // len(AA_ALPHABET)
            continue
        if source == "blosum62":
            matrices[source] = encode_blosum62_sequences(sequences, expected_length=expected_sequence_length)
            source_missing_counts[source] = 0
            computed_len = matrices[source].shape[1] // len(AA_ALPHABET)
            continue
        if source == TABULAR_FEATURE_SOURCE:
            if tabular_matrix is None:
                raise ValueError("tabular feature source requested but no tabular_matrix supplied")
            matrices[source] = np.asarray(tabular_matrix, dtype=np.float32)
            source_missing_counts[source] = 0
            continue
        if embedding_banks is None or source not in embedding_banks:
            raise ValueError(f"Missing embedding bank for feature source '{source}'")
        matrix, missing = assemble_single_embedding(sequences, embedding_banks[source])
        matrices[source] = matrix
        missing_any |= missing
        source_missing_counts[source] = int(missing.sum())

    if computed_len is None:
        try:
            computed_len = infer_sequence_length(sequences, expected_length=expected_sequence_length)
        except ValueError:
            computed_len = expected_sequence_length or 0
    return matrices, missing_any, source_missing_counts, computed_len


def transform_feature_mode(
    X_by_source_raw: Dict[str, np.ndarray],
    wt_index: Optional[int],
    feature_mode: str,
    wt_by_source: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    if feature_mode not in {"raw", "delta", "raw_plus_delta"}:
        raise ValueError(f"Unsupported feature mode: {feature_mode}")
    if feature_mode != "raw" and wt_index is None and wt_by_source is None:
        raise ValueError("WT reference is required for delta-based feature modes")
    transformed: Dict[str, np.ndarray] = {}
    for source, matrix in X_by_source_raw.items():
        if feature_mode == "raw":
            transformed[source] = matrix
            continue
        wt_vec = wt_by_source[source] if (wt_by_source is not None and source in wt_by_source) else matrix[wt_index]
        delta = matrix - wt_vec
        if feature_mode == "delta":
            transformed[source] = delta.astype(np.float32, copy=False)
        else:
            transformed[source] = np.concatenate([matrix, delta], axis=1).astype(np.float32, copy=False)
    return transformed


def all_nonempty_feature_subsets(names: Sequence[str]) -> List[Tuple[str, ...]]:
    clean = [normalize_feature_source(n) for n in names]
    subsets: List[Tuple[str, ...]] = []
    for size in range(1, len(clean) + 1):
        subsets.extend(combinations(clean, size))
    return subsets


# --------------------------------------------------------------------------- #
# Stratified CV
# --------------------------------------------------------------------------- #


def make_cv_splits(
    n_samples: int,
    y: np.ndarray,
    groups: np.ndarray,
    cv_splits: int,
    random_state: int,
    strategy: str = "auto",
    n_repeats: int = 1,
    holdout_fraction: float = 0.2,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], bool]:
    y = np.asarray(y)
    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))
    # smallest class count caps the number of stratified folds
    min_class = int(np.min(np.bincount(y))) if len(y) else 0
    k = max(2, min(cv_splits, max(2, min_class)))

    if strategy == "holdout":
        idx = np.arange(n_samples)
        try:
            train_idx, valid_idx = train_test_split(
                idx, test_size=holdout_fraction, random_state=random_state, stratify=y
            )
        except Exception:
            train_idx, valid_idx = train_test_split(idx, test_size=holdout_fraction, random_state=random_state)
        return [(np.sort(train_idx), np.sort(valid_idx))], False

    if strategy == "repeated":
        splitter = RepeatedStratifiedKFold(n_splits=k, n_repeats=max(1, n_repeats), random_state=random_state)
        return list(splitter.split(np.zeros((n_samples, 1)), y)), False

    if strategy in {"auto", "group"} and HAS_SGKF and n_groups >= k:
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=random_state)
        return list(splitter.split(np.zeros((n_samples, 1)), y, groups=groups)), True

    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    return list(splitter.split(np.zeros((n_samples, 1)), y)), False


def predict_proba_aligned(estimator: BaseEstimator, X: np.ndarray, n_classes: int) -> np.ndarray:
    """Return an (n, n_classes) probability matrix aligned to global class ids 0..K-1,
    filling columns for classes the fitted estimator actually saw."""
    proba = np.asarray(estimator.predict_proba(X), dtype=float)
    classes_ = np.asarray(getattr(estimator, "classes_", np.arange(proba.shape[1])), dtype=int)
    if proba.shape[1] == n_classes and np.array_equal(classes_, np.arange(n_classes)):
        return proba
    full = np.zeros((proba.shape[0], n_classes), dtype=float)
    for col, cls in enumerate(classes_):
        if 0 <= int(cls) < n_classes:
            full[:, int(cls)] = proba[:, col]
    row_sums = full.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return full / row_sums


def cross_val_predict_proba(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    n_classes: int,
    classes: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, List[Dict[str, float]]]:
    n = len(y)
    proba_sum = np.zeros((n, n_classes), dtype=float)
    counts = np.zeros(n, dtype=int)
    fold_ids = np.full(n, -1, dtype=int)
    fold_rows: List[Dict[str, float]] = []

    for fold_number, (train_idx, valid_idx) in enumerate(splits, start=1):
        model = clone(estimator)
        model.fit(X[train_idx], y[train_idx])
        p = predict_proba_aligned(model, X[valid_idx], n_classes)
        proba_sum[valid_idx] += p
        counts[valid_idx] += 1
        fold_ids[valid_idx] = fold_number
        m = compute_metrics(classes[y[valid_idx]], p, classes)
        m["fold"] = fold_number
        m["n_train"] = int(len(train_idx))
        m["n_valid"] = int(len(valid_idx))
        fold_rows.append(m)

    seen = counts > 0
    oof_proba = np.full((n, n_classes), np.nan, dtype=float)
    oof_proba[seen] = proba_sum[seen] / counts[seen][:, None]
    if not seen.any():
        raise RuntimeError("No validation predictions were produced")

    metrics = compute_metrics(classes[y[seen]], oof_proba[seen], classes)
    return metrics, oof_proba, fold_ids, fold_rows


def prepare_supervised_dataframe(
    df: pd.DataFrame,
    seq_col: str,
    target_col: str,
    replicate_policy: str = "majority_by_sequence",
    id_col: Optional[str] = None,
) -> pd.DataFrame:
    """Clean + (optionally) aggregate replicate rows for a categorical target.

    Aggregation uses the majority label per sequence (ties -> first seen).
    """
    work = df.copy()
    work[seq_col] = work[seq_col].astype(str).str.strip()
    work = work[work[target_col].notna() & work[seq_col].notna()].copy()
    work[target_col] = work[target_col].astype(str)
    work = work[work[seq_col].str.len() > 0].reset_index(drop=True)

    if replicate_policy in {"keep_rows"}:
        work["is_aggregated_row"] = False
        work["n_source_rows"] = 1
        if id_col and id_col in work.columns:
            work["source_ids"] = work[id_col].astype(str)
        return work

    if replicate_policy not in {"majority_by_sequence"}:
        raise ValueError(f"Unsupported replicate policy: {replicate_policy}")

    def _majority(s: pd.Series):
        vc = s.value_counts()
        return vc.index[0]

    grouped = work.groupby(seq_col, sort=False, dropna=False)
    agg_label = grouped[target_col].agg(_majority)
    counts = grouped.size()
    base = work.drop_duplicates(subset=[seq_col], keep="first").reset_index(drop=True)
    base[target_col] = base[seq_col].map(agg_label).astype(str)
    base["is_aggregated_row"] = True
    base["n_source_rows"] = base[seq_col].map(counts).astype(int)
    if id_col and id_col in work.columns:
        if len(work) <= 200_000:
            ids = grouped[id_col].agg(lambda s: "|".join(map(str, s)))
            base["source_ids"] = base[seq_col].map(ids)
        else:
            base["source_ids"] = base[id_col].astype(str)
    return base


def proba_score_dataframe(
    df: pd.DataFrame,
    run_name: str,
    proba: np.ndarray,
    classes: np.ndarray,
    positive_class: Optional[str] = None,
    fold_ids: Optional[np.ndarray] = None,
    true_labels: Optional[np.ndarray] = None,
    ensemble_std: Optional[np.ndarray] = None,
    seen_in_train: Optional[np.ndarray] = None,
    missing_any_feature: Optional[np.ndarray] = None,
    rank: bool = False,
) -> pd.DataFrame:
    """Attach predicted label, per-class probabilities, confidence, entropy and an
    optional ranking by the positive class probability."""
    out = df.copy()
    out["run_name"] = run_name
    proba = np.asarray(proba, dtype=float)
    n_classes = len(classes)
    valid = ~np.all(np.isnan(proba), axis=1)
    pred_idx = np.full(len(out), -1, dtype=int)
    pred_idx[valid] = np.argmax(np.nan_to_num(proba[valid], nan=-1.0), axis=1)
    out["pred_label"] = [str(classes[i]) if i >= 0 else "" for i in pred_idx]
    out["confidence"] = np.where(valid, np.nanmax(proba, axis=1), np.nan)
    out["entropy"] = np.where(valid, predictive_entropy(np.nan_to_num(proba, nan=1.0 / max(1, n_classes))), np.nan)
    for j, cls in enumerate(classes):
        out[f"proba_{cls}"] = proba[:, j]
    if true_labels is not None:
        out["y_true"] = [str(t) for t in true_labels]
        out["correct"] = out["y_true"] == out["pred_label"]
    if fold_ids is not None:
        out["fold"] = np.asarray(fold_ids).astype(int)
    out["pred_ensemble_std"] = np.nan if ensemble_std is None else np.asarray(ensemble_std).ravel()
    if seen_in_train is not None:
        out["seen_in_train"] = np.asarray(seen_in_train).astype(bool)
    if missing_any_feature is not None:
        out["missing_any_feature"] = np.asarray(missing_any_feature).astype(bool)

    # ranking score: probability of the positive/target class (default last class)
    pos = positive_class if (positive_class is not None and positive_class in [str(c) for c in classes]) else str(classes[-1])
    out["score"] = out[f"proba_{pos}"]
    out["ranked_class"] = pos
    if rank:
        valid_mask = out["score"].notna()
        ranks = pd.Series(np.nan, index=out.index, dtype=float)
        ranks.loc[valid_mask] = out.loc[valid_mask, "score"].rank(method="first", ascending=False)
        out["rank"] = ranks
        out = out.sort_values(["rank"], na_position="last").reset_index(drop=True)
    return out


@dataclass
class FittedRunModel:
    run_name: str
    trial_number: int
    score: float
    metric_name: str
    model_name: str
    feature_sources: Tuple[str, ...]
    feature_mode: str
    estimator: BaseEstimator
    classes: Tuple[str, ...]
    n_classes: int
    train_sequences: Tuple[str, ...] = field(default_factory=tuple)
    expected_sequence_length: Optional[int] = None
    wt_by_source: Dict[str, np.ndarray] = field(default_factory=dict)

    def build_matrix(self, X_by_source: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([X_by_source[s] for s in self.feature_sources], axis=1)

    def predict_proba(self, X_by_source: Dict[str, np.ndarray]) -> np.ndarray:
        matrix = self.build_matrix(X_by_source)
        return predict_proba_aligned(self.estimator, matrix, self.n_classes)


@dataclass
class FittedEnsemble:
    run_name: str
    fitted_models: List[FittedRunModel]
    n_classes: int

    def predict_proba(self, X_by_mode: Dict[str, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fitted_models:
            raise ValueError("Ensemble has no fitted models")
        stack = []
        for model in self.fitted_models:
            stack.append(model.predict_proba(X_by_mode[model.feature_mode]))
        arr = np.stack(stack, axis=0)  # (members, n, K)
        mean = arr.mean(axis=0)
        # uncertainty = mean across classes of the per-class std across members
        std = arr.std(axis=0, ddof=0).mean(axis=1)
        return mean, std
