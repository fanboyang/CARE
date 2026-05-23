#!/usr/bin/env python3
"""Prepare benchmark split folders for CARE."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DATASETS = ("cn15k", "nl27k", "ppi5k")
SPLIT_FILES = ("train.tsv", "val.tsv", "test.tsv", "test_with_neg.tsv", "softlogic.tsv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy UnKGCP-style benchmark splits into data/benchmarks.")
    parser.add_argument("--source-root", required=True, help="Root folder containing extracted UnKGCP/UKGE data.")
    parser.add_argument("--output-root", default="data/benchmarks", help="Destination benchmark folder.")
    return parser.parse_args()


def find_dataset_dir(source_root: Path, dataset: str) -> Path:
    candidates: list[Path] = []
    for path in source_root.rglob("train.tsv"):
        parent = path.parent
        name = parent.name.lower()
        if name == dataset or name == f"{dataset}_no_psl":
            candidates.append(parent)
    if not candidates:
        raise FileNotFoundError(f"cannot find {dataset}/train.tsv under {source_root}")
    return sorted(candidates, key=lambda item: (item.name.lower() != dataset, len(str(item))))[0]


def copy_splits(source_dir: Path, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in SPLIT_FILES:
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)
            copied.append(name)
    if "train.tsv" not in copied:
        raise FileNotFoundError(f"{source_dir} does not contain train.tsv")
    return copied


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    for dataset in DATASETS:
        src = find_dataset_dir(source_root, dataset)
        copied = copy_splits(src, output_root / dataset)
        print(f"{dataset}: {src} -> {output_root / dataset} ({', '.join(copied)})")


if __name__ == "__main__":
    main()
