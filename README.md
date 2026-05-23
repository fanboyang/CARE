# CARE Release

This folder contains the public release code for CARE, a context-anchored
asymmetric conformal interval method for uncertain knowledge graph confidence
scores.

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
│   ├── prepare_data_splits.py
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

## Data Preparation

Download the uncertain-KG benchmark files from the UnKGCP data link:

```text
https://1drv.ms/u/c/53E5DF60F4E3FFE8/EYFHc7ZRt-NAturvecIjUBMBG2xXgD3Ly0PNhIbzeOMONQ?e=hyC3Dr
```

The original UKGE base data are also available from:

```text
https://drive.google.com/file/d/1UJQ8hnqPGv1O9pYglfNF5lY_sgDQkleS/view?usp=sharing
```

Extract the downloaded files outside git, for example under
`data_sources/unkgcp/`, then prepare the release layout:

```bash
python scripts/prepare_data_splits.py \
  --source-root data_sources/unkgcp \
  --output-root data/benchmarks
```

The prepared training graphs should appear as
`data/benchmarks/{cn15k,nl27k,ppi5k}/train.tsv`. See `data/README.md` for the
expected split layout and `score_exports/README.md` for fixed score files.

## Run The Paper-Style 10-Seed Evaluation

```bash
python scripts/run_10seed.py
```

By default this runs all three datasets, all three score models, and seeds
`0…9`. For seed `i`, the score export seed and the calibration split seed are
both fixed to `i`.

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
