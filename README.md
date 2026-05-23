# CARE Release

This folder contains the public release code for CARE. The release keeps only
the interval construction path used for the reported experiments. It consumes
fixed confidence-score exports from scoring models and converts each score into
a context-anchored asymmetric conformal interval.

## Layout

```text
care_release/
├── README.md
├── requirements.txt
├── CARE/
│   └── interval.py
├── configs/
│   └── CARE_default/
│       └── global.json
├── scripts/
│   └── run_10seed.py
├── data/
│   └── README.md
├── score_exports/
│   └── README.md
└── results/
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

CARE supports two explicit nearest-neighbor backends:

- `cuda`: FAISS GPU exact L2 search. Install `faiss-gpu` separately.
- `cpu`: batched NumPy exact L2 search. No FAISS package required.

## Required Inputs

CARE needs two input families:

1. Training graph splits under `data/benchmarks/<dataset>/train.tsv`.
2. Fixed score exports under `score_exports/<score_model>/<dataset>/seed<id>/`.

The release does not include raw datasets, pretrained checkpoints, score-export
matrices, logs, or cached nearest-neighbor artifacts. See `data/README.md` and
`score_exports/README.md` for file formats.

## Run The Paper-Style 10-Seed Evaluation

```bash
python scripts/run_10seed.py
```

By default this runs all three datasets, all three score models, and seeds
`0…9`. For seed `i`, the score export seed and the calibration split seed are
both fixed to `i`. The default KNN backend is `cuda`; pass `--knn-backend cpu`
on machines without CUDA FAISS. Narrower runs:

```bash
python scripts/run_10seed.py --datasets cn15k --score-models ukge
python scripts/run_10seed.py --datasets ppi5k --knn-backend cpu
```

Outputs are written to:

```text
results/CARE_10seed/
├── per_seed/
└── summary.json
```

## Release Notes

This release contains only method code, configuration, and input-format
documentation. It intentionally excludes local paths, generated manuscript
artifacts, raw datasets, checkpoints, logs, cached KNN files, and non-release
experiment scripts.
