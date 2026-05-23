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
from care.interval import DATASETS, SCORE_MODELS, SUBSETS, evaluate_cell, load_config, write_json  # noqa: E402


def _csv_list(text: str, choices: tuple[str, ...]) -> tuple[str, ...]:
    items = tuple(item.strip() for item in text.split(",") if item.strip())
    bad = [item for item in items if item not in choices]
    if bad:
        raise ValueError(f"unsupported values: {bad}")
    return items


def _seed_list(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CARE on the 10-seed evaluation matrix.")
    parser.add_argument("--datasets", default="cn15k,ppi5k,nl27k")
    parser.add_argument("--score-models", default="ukge,passleaf,beurre")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--target-coverage", type=float, default=0.90)
    parser.add_argument("--score-root", default="score_exports")
    parser.add_argument("--config-dir", default="configs/care_default")
    parser.add_argument("--output-dir", default="results/care_10seed")
    return parser.parse_args()


def _summarize(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for score_model in SCORE_MODELS:
        for dataset in DATASETS:
            cell = [item for item in records if item["score_model"] == score_model and item["dataset"] == dataset]
            if not cell:
                continue
            for subset in SUBSETS:
                cov = np.asarray([item["subsets"][subset]["metrics"]["coverage"] for item in cell], dtype=float)
                sharp = np.asarray([item["subsets"][subset]["metrics"]["sharpness"] for item in cell], dtype=float)
                rows.append(
                    {
                        "score_model": score_model,
                        "dataset": dataset,
                        "subset": subset,
                        "seeds": int(len(cell)),
                        "coverage_mean": float(np.mean(cov)),
                        "coverage_var": float(np.var(cov, ddof=1)) if len(cov) > 1 else 0.0,
                        "sharpness_mean": float(np.mean(sharp)),
                        "sharpness_var": float(np.var(sharp, ddof=1)) if len(sharp) > 1 else 0.0,
                    }
                )
    return rows


def main() -> None:
    args = parse_args()
    datasets = _csv_list(args.datasets, DATASETS)
    score_models = _csv_list(args.score_models, SCORE_MODELS)
    seeds = _seed_list(args.seeds)
    cfg = load_config(args.config_dir, args.target_coverage)
    records = []
    for score_model in score_models:
        for dataset in datasets:
            for seed in seeds:
                result = evaluate_cell(args.score_root, score_model, dataset, seed, replace(cfg, split_seed=int(seed)))
                out = Path(args.output_dir) / "per_seed" / score_model / dataset / f"seed{seed}.json"
                write_json(out, result)
                records.append(result)
                print(f"finished score_model={score_model} dataset={dataset} seed={seed}")
    summary = {"method": "CARE", "config": cfg.__dict__, "summary": _summarize(records)}
    path = write_json(Path(args.output_dir) / "summary.json", summary)
    print(path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
