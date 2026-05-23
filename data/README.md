# Data Directory

This release does not redistribute raw uncertain-KG datasets.

Prepare the benchmark splits with this directory structure:

```text
data/
└── benchmarks/
    ├── cn15k/
    │   └── train.tsv
    ├── nl27k/
    │   └── train.tsv
    └── ppi5k/
        └── train.tsv
```

Each `train.tsv` file must contain at least four tab-separated columns:

```text
head_id    relation_id    tail_id    confidence_label
```

The CARE feature builder only uses `train.tsv` to compute graph-structural
features. Calibration and test labels are loaded from `score_exports/`.
