#!/usr/bin/env python3
"""PLM-Classifier command line: ``plm-classifier {train,predict,embed,gui,list}``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional


def _cmd_train(args) -> int:
    from .config import RunConfig
    from .train import run_training

    run_training(RunConfig.from_yaml(args.config))
    return 0


def _cmd_predict(args) -> int:
    from .predict import score_candidates_from_run

    if not args.run_dir or not args.candidate_csv:
        raise SystemExit("--run-dir and --candidate-csv are required")
    score_candidates_from_run(
        run_dir=args.run_dir, candidate_csv=args.candidate_csv, predict_seq_col=args.predict_seq_col,
        candidate_embedding_dir=args.candidate_embedding_dir, top_n=args.top_n, positive_class=args.positive_class,
    )
    return 0


def _cmd_embed(args) -> int:
    from .embeddings.extract import extract_from_csv

    path, n_new, n_cached = extract_from_csv(
        plm_name=args.plm, input_csv=args.input_csv, seq_col=args.seq_col,
        output_npz=args.output_npz, batch_size=args.batch_size, force_cpu=args.cpu,
    )
    print(f"Wrote {path}: {n_new} computed, {n_cached} reused from cache")
    return 0


def _cmd_gui(args) -> int:
    import subprocess

    app = Path(__file__).resolve().parent.parent / "app.py"
    if not app.exists():
        raise SystemExit(f"GUI app not found at {app}")
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])


def _cmd_list(args) -> int:
    from .registry import MODEL_REGISTRY, PLM_REGISTRY, available_models, available_plms

    print("Classifiers (available):", ", ".join(available_models()))
    print("Classifiers (all):", ", ".join(MODEL_REGISTRY))
    print("pLMs (available):", ", ".join(available_plms()))
    print("pLMs (all):", ", ".join(PLM_REGISTRY))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="plm-classifier", description="General sequence -> class label classification framework")
    sub = p.add_subparsers(dest="command", required=True)

    pt = sub.add_parser("train", help="Train from a RunConfig YAML")
    pt.add_argument("config")
    pt.set_defaults(func=_cmd_train)

    pp = sub.add_parser("predict", help="Score/rank candidates from a saved run")
    pp.add_argument("--run-dir", required=True)
    pp.add_argument("--candidate-csv", required=True)
    pp.add_argument("--predict-seq-col")
    pp.add_argument("--candidate-embedding-dir")
    pp.add_argument("--positive-class", help="Class whose probability ranks candidates (default: run's positive_class)")
    pp.add_argument("--top-n", type=int, default=100)
    pp.set_defaults(func=_cmd_predict)

    pe = sub.add_parser("embed", help="Extract (and cache) pLM embeddings")
    pe.add_argument("--plm", required=True)
    pe.add_argument("--input-csv", required=True)
    pe.add_argument("--seq-col", required=True)
    pe.add_argument("--output-npz", required=True)
    pe.add_argument("--batch-size", type=int, default=8)
    pe.add_argument("--cpu", action="store_true")
    pe.set_defaults(func=_cmd_embed)

    pg = sub.add_parser("gui", help="Launch the Streamlit GUI")
    pg.set_defaults(func=_cmd_gui)

    pl = sub.add_parser("list", help="List available classifiers and pLMs")
    pl.set_defaults(func=_cmd_list)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
