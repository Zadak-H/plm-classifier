#!/usr/bin/env python3
"""Assemble the per-run classification report: metrics + plots -> json + html."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .core import load_json, save_json
from .metrics import compute_metrics
from . import plots


def _oof_metrics(out_dir: Path, classes: List[str]) -> Dict[str, float]:
    df = pd.read_csv(out_dir / "oof_predictions.csv")
    df = df[df["y_true"].notna()]
    proba = np.column_stack([df[f"proba_{c}"].to_numpy(dtype=float) for c in classes])
    return compute_metrics(df["y_true"].astype(str).to_numpy(), proba, np.array([str(c) for c in classes]))


def build_report(out_dir: str | Path, metric: str = "roc_auc") -> Path:
    out_dir = Path(out_dir)
    summary = load_json(out_dir / "run_summary.json") if (out_dir / "run_summary.json").exists() else {}
    classes = [str(c) for c in summary.get("classes", [])]
    metrics = _oof_metrics(out_dir, classes) if classes else {}

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    images: List[str] = []

    def _try(fn, *args, **kwargs):
        try:
            path = fn(*args, **kwargs)
            if path is not None:
                images.append(Path(path).name)
        except Exception as exc:  # pragma: no cover
            print(f"[warn] plot {fn.__name__} skipped: {exc}")

    oof = out_dir / "oof_predictions.csv"
    if classes:
        _try(plots.confusion_matrix_plot, oof, classes, plots_dir / "confusion_matrix.png")
        _try(plots.roc_curve_plot, oof, classes, plots_dir / "roc_curve.png")
        _try(plots.pr_curve_plot, oof, classes, plots_dir / "pr_curve.png")
        _try(plots.calibration_plot, oof, classes, plots_dir / "calibration.png")
    if (out_dir / "search_history.csv").exists():
        _try(plots.model_comparison_bar, out_dir / "search_history.csv", plots_dir / "model_comparison.png", metric)

    report = {
        "task": "classification", "run_name": summary.get("run_name", out_dir.name),
        "primary_metric": metric, "oof_metrics": metrics, "classes": classes,
        "class_counts": summary.get("class_counts"), "positive_class": summary.get("positive_class"),
        "best_model_name": summary.get("best_model_name"), "best_feature_subset": summary.get("best_feature_subset"),
        "best_feature_mode": summary.get("best_feature_mode"), "size_tier": summary.get("size_tier"),
        "cv_strategy": summary.get("cv_strategy"), "plots": images,
    }
    save_json(out_dir / "run_report.json", report)
    _write_html(out_dir, report, metrics)
    return out_dir / "run_report.html"


def _write_html(out_dir: Path, report: dict, metrics: Dict[str, float]) -> None:
    rows = "".join(f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in metrics.items() if isinstance(v, (int, float)))
    imgs = "".join(f'<div class="card"><img src="plots/{Path(n).name}"></div>' for n in report.get("plots", []))
    classes = report.get("classes") or []
    counts = report.get("class_counts") or []
    cls_line = ", ".join(f"{c} (n={n})" for c, n in zip(classes, counts)) if counts else ", ".join(classes)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>PLM-Classifier report: {report['run_name']}</title>
<style>
body{{font-family:system-ui,Arial,sans-serif;margin:24px;color:#222}}
h1{{margin-bottom:0}} .sub{{color:#666;margin-top:4px}}
table{{border-collapse:collapse;margin:16px 0}} td,th{{border:1px solid #ccc;padding:6px 12px;text-align:left}}
.card{{display:inline-block;margin:10px;vertical-align:top}} img{{max-width:520px;border:1px solid #eee;border-radius:6px}}
.kv{{background:#f6f8fa;padding:10px 14px;border-radius:6px;display:inline-block;margin:4px}}
</style></head><body>
<h1>PLM-Classifier run report</h1>
<div class="sub">{report['run_name']} &middot; tier: {report.get('size_tier')} &middot; CV: {report.get('cv_strategy')}</div>
<p>
<span class="kv">best model: <b>{report.get('best_model_name')}</b></span>
<span class="kv">features: <b>{'+'.join(report.get('best_feature_subset') or [])}</b> ({report.get('best_feature_mode')})</span>
<span class="kv">classes: <b>{cls_line}</b></span>
<span class="kv">ranked by P(<b>{report.get('positive_class')}</b>)</span>
</p>
<h2>OOF metrics</h2>
<table><tr><th>metric</th><th>value</th></tr>{rows}</table>
<h2>Plots</h2>
<div>{imgs}</div>
</body></html>"""
    (out_dir / "run_report.html").write_text(html, encoding="utf-8")
