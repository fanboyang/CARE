# Score Export Directory

CARE evaluates fixed confidence-score exports generated from the same benchmark
splits prepared under `data/benchmarks/`.

Use this directory structure:

```text
score_exports/
└── <score_model>/
    └── <dataset>/
        └── seed<id>/
            ├── calibration_val.csv
            ├── calibration_neg.csv
            ├── test_pos.csv
            ├── test_neg.csv
            └── test_with_neg.csv
```

Supported score models:

```text
ukge, passleaf, beurre
```

Supported datasets:

```text
cn15k, nl27k, ppi5k
```

Each CSV must contain these columns:

```text
h,r,t,label,score,source_split
```

`score` is the fixed model confidence score. `label` is the gold confidence
value used for conformal calibration and evaluation. Put one complete set of
CSV files under every `score_exports/<score_model>/<dataset>/seed<id>/` folder
used by `scripts/run_10seed.py`.
