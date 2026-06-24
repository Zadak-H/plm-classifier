#!/usr/bin/env python3
"""Classification metrics for the PLM-Classifier framework.

Handles binary and multiclass. Probability-aware metrics (ROC-AUC, PR-AUC,
log-loss) use the predicted class-probability matrix; the rest use hard labels.
All functions are defensive so a degenerate Optuna trial never crashes the search.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Metrics usable as a single scalar optimization objective.
METRIC_CHOICES = (
    "roc_auc",
    "pr_auc",
    "f1",
    "f1_macro",
    "accuracy",
    "balanced_accuracy",
    "mcc",
    "log_loss",
)

_MINIMIZE = {"log_loss"}


def metric_direction(metric_name: str) -> str:
    return "minimize" if metric_name in _MINIMIZE else "maximize"


def sort_descending_for_metric(metric_name: str) -> bool:
    return metric_direction(metric_name) == "maximize"


def _labels_from_proba(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return classes[np.argmax(proba, axis=1)]


def compute_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: np.ndarray,
) -> Dict[str, float]:
    """Full classification bundle from a probability matrix.

    ``y_true`` are class labels (any hashable), ``proba`` is (n, n_classes) aligned
    to ``classes`` (sorted unique training labels).
    """
    y_true = np.asarray(y_true).ravel()
    proba = np.asarray(proba, dtype=float)
    classes = np.asarray(classes)
    n_classes = len(classes)
    y_pred = _labels_from_proba(proba, classes)

    out: Dict[str, float] = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    try:
        out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        out["balanced_accuracy"] = 0.0
    out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    out["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    out["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    try:
        out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        out["mcc"] = 0.0

    is_binary = n_classes == 2
    # binary "f1" = positive-class F1 (positive = classes[1]); else macro
    if is_binary:
        out["f1"] = float(f1_score(y_true, y_pred, pos_label=classes[1], zero_division=0))
    else:
        out["f1"] = out["f1_macro"]

    # probability metrics
    try:
        if is_binary:
            out["roc_auc"] = float(roc_auc_score((y_true == classes[1]).astype(int), proba[:, 1]))
            out["pr_auc"] = float(average_precision_score((y_true == classes[1]).astype(int), proba[:, 1]))
        else:
            y_onehot = np.zeros((len(y_true), n_classes), dtype=int)
            cls_index = {c: i for i, c in enumerate(classes)}
            for i, yt in enumerate(y_true):
                j = cls_index.get(yt)
                if j is not None:
                    y_onehot[i, j] = 1
            out["roc_auc"] = float(roc_auc_score(y_onehot, proba, average="macro", multi_class="ovr"))
            out["pr_auc"] = float(average_precision_score(y_onehot, proba, average="macro"))
    except Exception:
        out["roc_auc"] = 0.0
        out["pr_auc"] = 0.0

    try:
        out["log_loss"] = float(log_loss(y_true, proba, labels=list(classes)))
    except Exception:
        out["log_loss"] = float("nan")

    return out


def predictive_entropy(proba: np.ndarray) -> np.ndarray:
    """Shannon entropy per row (nats) — a simple confidence/uncertainty signal."""
    p = np.clip(np.asarray(proba, dtype=float), 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)
