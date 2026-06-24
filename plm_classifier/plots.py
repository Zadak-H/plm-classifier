#!/usr/bin/env python3
"""Classification plots for the run report. Each function saves a PNG, returns its path."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def _proba_matrix(df: pd.DataFrame, classes: List[str]) -> np.ndarray:
    return np.column_stack([df[f"proba_{c}"].to_numpy(dtype=float) for c in classes])


def confusion_matrix_plot(oof_csv, classes: List[str], out_png) -> Path:
    df = pd.read_csv(oof_csv)
    df = df[df["y_true"].notna() & (df["pred_label"].astype(str) != "")]
    yt = df["y_true"].astype(str); yp = df["pred_label"].astype(str)
    cm = confusion_matrix(yt, yp, labels=[str(c) for c in classes])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(cm, display_labels=[str(c) for c in classes]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion matrix (OOF)")
    plt.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)
    return Path(out_png)


def roc_curve_plot(oof_csv, classes: List[str], out_png) -> Optional[Path]:
    df = pd.read_csv(oof_csv)
    df = df[df["y_true"].notna()]
    classes = [str(c) for c in classes]
    proba = _proba_matrix(df, classes)
    yt = df["y_true"].astype(str).to_numpy()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    try:
        if len(classes) == 2:
            pos = classes[1]
            fpr, tpr, _ = roc_curve((yt == pos).astype(int), proba[:, 1])
            ax.plot(fpr, tpr, lw=2, label=f"AUC={auc(fpr, tpr):.3f}")
        else:
            for j, c in enumerate(classes):
                fpr, tpr, _ = roc_curve((yt == c).astype(int), proba[:, j])
                ax.plot(fpr, tpr, lw=1.5, label=f"{c} (AUC={auc(fpr, tpr):.2f})")
    except Exception:
        plt.close(fig); return None
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC (OOF)"); ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)
    return Path(out_png)


def pr_curve_plot(oof_csv, classes: List[str], out_png) -> Optional[Path]:
    df = pd.read_csv(oof_csv)
    df = df[df["y_true"].notna()]
    classes = [str(c) for c in classes]
    proba = _proba_matrix(df, classes)
    yt = df["y_true"].astype(str).to_numpy()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    try:
        targets = [(classes[1], 1)] if len(classes) == 2 else list(zip(classes, range(len(classes))))
        for c, j in targets:
            prec, rec, _ = precision_recall_curve((yt == c).astype(int), proba[:, j])
            ax.plot(rec, prec, lw=1.8, label=f"{c} (AP={auc(rec, prec):.2f})")
    except Exception:
        plt.close(fig); return None
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall (OOF)"); ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)
    return Path(out_png)


def calibration_plot(oof_csv, classes: List[str], out_png) -> Optional[Path]:
    classes = [str(c) for c in classes]
    if len(classes) != 2:
        return None
    df = pd.read_csv(oof_csv)
    df = df[df["y_true"].notna()]
    p = df[f"proba_{classes[1]}"].to_numpy(dtype=float)
    y = (df["y_true"].astype(str).to_numpy() == classes[1]).astype(int)
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    xs, ys = [], []
    for b in range(10):
        m = idx == b
        if m.sum() > 0:
            xs.append(p[m].mean()); ys.append(y[m].mean())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="black", label="ideal")
    ax.plot(xs, ys, "-o", color="#E6550D", label="observed")
    ax.set_xlabel("Mean predicted prob"); ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration (positive={classes[1]})"); ax.legend(); ax.grid(True, linestyle=":")
    plt.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)
    return Path(out_png)


def model_comparison_bar(search_history_csv, out_png, metric="roc_auc") -> Optional[Path]:
    df = pd.read_csv(search_history_csv)
    if "model_name" not in df.columns or df.empty:
        return None
    best = df.groupby("model_name")["value"].max().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(best)), 4.5))
    ax.bar(best.index.astype(str), best.values, color="#2C7FB8")
    ax.set_ylabel(f"best {metric} (OOF)"); ax.set_title("Best score by classifier")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)
    return Path(out_png)
