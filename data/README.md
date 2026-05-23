# Data Directory

Download the uncertain-KG benchmark files from the UnKGCP data link:

```text
https://1drv.ms/u/c/53E5DF60F4E3FFE8/EYFHc7ZRt-NAturvecIjUBMBG2xXgD3Ly0PNhIbzeOMONQ?e=hyC3Dr
```

The original UKGE base data are also available from:

```text
https://drive.google.com/file/d/1UJQ8hnqPGv1O9pYglfNF5lY_sgDQkleS/view?usp=sharing
```

Place the extracted files outside git, for example:

```text
data_sources/
└── unkgcp/
    ├── cn15k/
    ├── nl27k/
    └── ppi5k/
```

Then run the split preparation script from the repository root:

```bash
python scripts/prepare_data_splits.py \
  --source-root data_sources/unkgcp \
  --output-root data/benchmarks
```

The prepared benchmark splits should use this layout:

```text
data/
└── benchmarks/
    ├── cn15k/
    │   ├── train.tsv
    │   ├── val.tsv
    │   ├── test.tsv
    │   └── test_with_neg.tsv
    ├── nl27k/
    │   ├── train.tsv
    │   ├── val.tsv
    │   ├── test.tsv
    │   └── test_with_neg.tsv
    └── ppi5k/
        ├── train.tsv
        ├── val.tsv
        ├── test.tsv
        └── test_with_neg.tsv
```

Each `train.tsv` file must contain at least four tab-separated columns:

```text
head_id    relation_id    tail_id    confidence_label
```

CARE uses `train.tsv` to compute graph-structural features. Calibration and
test confidence labels for the interval experiments are loaded from
`score_exports/`.
