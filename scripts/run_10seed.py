#!/usr/bin/env python3
"""Run the released 10-seed CARE evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from CARE.interval import (  # noqa: E402
    DATASETS,
    KNN_BACKENDS,
    PAPER_SEEDS,
    SCORE_MODELS,
    SUBSETS,
    evaluate_cell,
    load_config,
    split_seed_for_score_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CARE on the 10-seed evaluation matrix.")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--score-models", nargs="+", choices=SCORE_MODELS, default=list(SCORE_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PAPER_SEEDS))
    parser.add_argument("--target-coverage", type=float, default=0.90)
    parser.add_argument("--score-root", default="score_exports")
    parser.add_argument("--config-dir", default="configs/CARE_default")
    parser.add_argument("--output-dir", default="results/CARE_10seed")
    parser.add_argument("--knn-backend", choices=KNN_BACKENDS, default=None)
    return parser.parse_args()


def _summarize(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for score_model in SCORE_MODELS:
        for dataset in DATASETS:
            cell = [r for r in records if r["score_model"] == score_model and r["dataset"] == dataset]
            if not cell:
                continue
            for subset in SUBSETS:
                cov = np.asarray([r["subsets"][subset]["metrics"]["coverage"] for r in cell], dtype=float)
                sharp = np.asarray([r["subsets"][subset]["metrics"]["sharpness"] for r in cell], dtype=float)
                rows.append({
                    "score_model": score_model,
                    "dataset": dataset,
                    "subset": subset,
                    "seeds": int(len(cell)),
                    "coverage_mean": float(np.mean(cov)),
                    "sharpness_mean": float(np.mean(sharp)),
                })
    return rows


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config_dir, args.target_coverage)
    knn_backend = args.knn_backend or cfg.knn_backend
    records = []
    for score_model in args.score_models:
        for dataset in args.datasets:
            for seed in args.seeds:
                run_cfg = replace(cfg, split_seed=split_seed_for_score_seed(seed), knn_backend=knn_backend)
                result = evaluate_cell(args.score_root, score_model, dataset, seed, run_cfg)
                out = Path(args.output_dir) / "per_seed" / score_model / dataset / f"seed{seed}.json"
                write_json(out, result)
                records.append(result)
                print(f"finished score_model={score_model} dataset={dataset} seed={seed}")
    summary = {
        "method": "CARE",
        "config": {**cfg.__dict__, "knn_backend": knn_backend, "split_seed_policy": "score_seed"},
        "summary": _summarize(records),
    }
    path = write_json(Path(args.output_dir) / "summary.json", summary)
    print(path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
