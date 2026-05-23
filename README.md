# CARE Release

This folder contains the anonymous release code for CARE. The release keeps only
the CARE interval construction path used for the reported experiments. It consumes fixed
confidence-score exports from scoring models and converts each score into a
context-anchored asymmetric conformal interval.

## Layout

```text
care_release/
├── README.md
├── requirements.txt
├── run_cn15k.py
├── run_nl27k.py
├── run_ppi5k.py
├── care/
│   └── interval.py
├── configs/
│   └── care_default/
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

CARE supports three explicit nearest-neighbor backends:

- `cuda`: FAISS GPU exact L2 search. Install CUDA FAISS separately.
- `cpu`: batched NumPy exact L2 search. This path needs no FAISS package.
- `mps`: PyTorch MPS exact L2 search for Apple Silicon. Install PyTorch separately.

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
`0,1,2,3,4,5,6,7,8,9`. For seed `i`, the score export seed and the
calibration split seed are both fixed to `i`. The default KNN backend is `cuda`; use
`--knn-backend cpu` or `--knn-backend mps` on machines without CUDA FAISS.

Useful narrower runs:

```bash
python scripts/run_10seed.py --datasets cn15k --score-models ukge
python scripts/run_10seed.py --datasets ppi5k --score-models ukge --knn-backend cpu
python run_cn15k.py --score-models ukge
python run_ppi5k.py
python run_nl27k.py
```

Outputs are written to:

```text
results/care_10seed/
├── per_seed/
└── summary.json
```

## Default Configuration

The default configuration is stored in `configs/care_default/global.json`.

```text
target coverage = 0.90
K = 200
rho = 0.90
kappa = 50
reference/conformal split = 60/40
paper 10-seed split policy = score seed i uses calibration split seed i
retrieval = exact L2 with cuda/cpu/mps backend
```

## Double-Blind Notes

This release contains only method code, configuration, and input-format
documentation. It intentionally excludes local paths, user names, generated
manuscript artifacts, raw datasets, checkpoints, logs, cached KNN files, and
non-release experiment scripts.
