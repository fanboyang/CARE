"""CARE interval evaluation with one CUDA FAISS retrieval path."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCORE_MODELS = ("ukge", "passleaf", "beurre")
DATASETS = ("cn15k", "ppi5k", "nl27k")
SUBSETS = ("positive", "negative", "combined")
SPLIT_FILES = {
    "cal_pos": "calibration_val.csv",
    "cal_neg": "calibration_neg.csv",
    "test_pos": "test_pos.csv",
    "test_neg": "test_neg.csv",
    "test_mix": "test_with_neg.csv",
}


@dataclass(frozen=True)
class CareConfig:
    target_coverage: float = 0.90
    k: int = 200
    rho: float = 0.90
    kappa: float = 50.0
    ref_fraction: float = 0.60
    split_seed: int = 0


def load_config(path: str | Path | None = None, target_coverage: float | None = None) -> CareConfig:
    if path is None:
        cfg = {}
    else:
        cfg_path = Path(path)
        if not cfg_path.is_absolute():
            cfg_path = ROOT / cfg_path
        cfg = json.loads((cfg_path / "global.json").read_text(encoding="utf-8"))
    params = cfg.get("care", cfg.get("global_params", {}))
    split = cfg.get("calibration", cfg.get("calibration_protocol", {}))
    return CareConfig(
        target_coverage=float(target_coverage if target_coverage is not None else cfg.get("target_coverage", 0.90)),
        k=int(params.get("k", 200)),
        rho=float(params.get("rho", params.get("local_quantile", 0.90))),
        kappa=float(params.get("kappa", params.get("shrink", 50.0))),
        ref_fraction=float(split.get("ref_fraction", split.get("reference_pool_fraction", 0.60))),
        split_seed=int(split.get("split_seed", 0)),
    )


def _rooted(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _q_high(values: np.ndarray, level: float) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("empty conformal score set")
    tau = min(1.0, math.ceil((values.size + 1) * float(level)) / values.size)
    return float(np.quantile(values, tau, method="higher"))


def _entropy(score: np.ndarray, dataset: str) -> np.ndarray:
    p = np.asarray(score, dtype=float)
    if dataset == "cn15k":
        p = 0.5 * p + 0.5
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)


def _read_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"h": str, "r": str, "t": str, "label": str, "score": float})
    needed = {"h", "r", "t", "label", "score"}
    missing = needed.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} misses columns: {sorted(missing)}")
    frame = frame.copy()
    frame["label_value"] = frame["label"].astype(float)
    frame["score"] = frame["score"].astype(float)
    return frame


def load_score_splits(score_root: str | Path, score_model: str, dataset: str, seed: int) -> dict[str, pd.DataFrame]:
    if score_model not in SCORE_MODELS:
        raise ValueError(f"unsupported score model: {score_model}")
    if dataset not in DATASETS:
        raise ValueError(f"unsupported dataset: {dataset}")
    base = _rooted(score_root) / score_model / dataset / f"seed{int(seed)}"
    if not base.exists():
        raise FileNotFoundError(f"missing score export directory: {base}")
    return {name: _read_frame(base / filename) for name, filename in SPLIT_FILES.items()}


@lru_cache(maxsize=None)
def _graph_stats(dataset: str) -> dict[str, object]:
    path = ROOT / "data" / "benchmarks" / dataset / "train.tsv"
    if not path.exists():
        raise FileNotFoundError(f"missing training graph: {path}")
    train = pd.read_csv(path, sep="\t", header=None, names=["h", "r", "t", "label"], usecols=[0, 1, 2, 3])
    h = train["h"].to_numpy(dtype=int)
    r = train["r"].to_numpy(dtype=int)
    t = train["t"].to_numpy(dtype=int)
    n_ent = int(max(h.max(initial=0), t.max(initial=0))) + 1
    n_rel = int(r.max(initial=0)) + 1
    degree = np.zeros(n_ent, dtype=np.float32)
    np.add.at(degree, h, 1.0)
    np.add.at(degree, t, 1.0)
    rel_freq = np.bincount(r, minlength=n_rel).astype(np.float32)
    adj = [set() for _ in range(n_ent)]
    edges = set()
    for left, right in zip(h, t):
        a, b = int(left), int(right)
        adj[a].add(b)
        adj[b].add(a)
        edges.add((a, b) if a <= b else (b, a))
    return {"degree": degree, "rel_freq": rel_freq, "adj": adj, "edges": edges}


def _counts(values: np.ndarray, ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids, dtype=int)
    out = np.zeros(len(ids), dtype=np.float32)
    mask = (0 <= ids) & (ids < len(values))
    out[mask] = values[ids[mask]]
    return out


def _common_neighbors(heads: np.ndarray, tails: np.ndarray, adj: list[set[int]]) -> np.ndarray:
    out = np.zeros(len(heads), dtype=np.float32)
    for i, (h, t) in enumerate(zip(heads, tails)):
        h, t = int(h), int(t)
        if h < 0 or t < 0 or h >= len(adj) or t >= len(adj):
            continue
        left, right = adj[h], adj[t]
        if len(left) > len(right):
            left, right = right, left
        out[i] = sum(1 for item in left if item in right)
    return out


def feature_matrix(frame: pd.DataFrame, dataset: str) -> np.ndarray:
    stats = _graph_stats(dataset)
    h = frame["h"].to_numpy(dtype=int)
    r = frame["r"].to_numpy(dtype=int)
    t = frame["t"].to_numpy(dtype=int)
    score = frame["score"].to_numpy(dtype=float).astype(np.float32)
    degree = stats["degree"]
    rel_freq = stats["rel_freq"]
    adj = stats["adj"]
    edges = stats["edges"]
    edge_flag = np.asarray(
        [1.0 if ((int(a), int(b)) if int(a) <= int(b) else (int(b), int(a))) in edges else 0.0 for a, b in zip(h, t)],
        dtype=np.float32,
    )
    scalars = np.column_stack(
        [
            score,
            _entropy(score, dataset).astype(np.float32),
            np.log1p(_counts(degree, h)),
            np.log1p(_counts(degree, t)),
            np.log1p(_counts(rel_freq, r)),
            edge_flag,
            np.log1p(_common_neighbors(h, t, adj)),
        ]
    ).astype(np.float32)
    rel_onehot = np.zeros((len(frame), len(rel_freq)), dtype=np.float32)
    valid = (0 <= r) & (r < len(rel_freq))
    rel_onehot[np.arange(len(frame))[valid], r[valid]] = 1.0
    return np.column_stack([scalars, rel_onehot]).astype(np.float32)


def _cuda_knn(reference: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
    import faiss

    reference = np.ascontiguousarray(reference, dtype=np.float32)
    target = np.ascontiguousarray(target, dtype=np.float32)
    gpu = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatL2(gpu, reference.shape[1])
    index.add(reference)
    _, idx = index.search(target, int(k))
    return idx.astype(np.int64, copy=False)


def _split_calibration(cal: pd.DataFrame, cfg: CareConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.split_seed)
    error = cal["score"].to_numpy(dtype=float) - cal["label_value"].to_numpy(dtype=float)
    sign = np.full(len(cal), "eq", dtype=object)
    sign[error > 1e-12] = "pos"
    sign[error < -1e-12] = "neg"
    ref_parts, holdout_parts = [], []
    work = cal.assign(_sign=sign).reset_index(drop=True)
    for _, group in work.groupby("_sign", sort=False):
        order = rng.permutation(len(group))
        cut = min(len(group) - 1, max(1, int(math.floor(len(group) * cfg.ref_fraction)))) if len(group) > 1 else 1
        ref_parts.append(group.iloc[order[:cut]].drop(columns=["_sign"]))
        holdout_parts.append(group.iloc[order[cut:]].drop(columns=["_sign"]))
    return pd.concat(ref_parts, ignore_index=True), pd.concat(holdout_parts, ignore_index=True)


def _residual(frame: pd.DataFrame, side: str) -> np.ndarray:
    error = frame["score"].to_numpy(dtype=float) - frame["label_value"].to_numpy(dtype=float)
    return np.maximum(error, 0.0) if side == "lower" else np.maximum(-error, 0.0)


def _scales(ref: pd.DataFrame, target: pd.DataFrame, dataset: str, cfg: CareConfig, side: str) -> np.ndarray:
    scaler = StandardScaler()
    ref_x = scaler.fit_transform(feature_matrix(ref, dataset)).astype(np.float32)
    target_x = scaler.transform(feature_matrix(target, dataset)).astype(np.float32)
    idx = _cuda_knn(ref_x, target_x, cfg.k)
    residual = _residual(ref, side)
    local = np.quantile(residual[idx], cfg.rho, axis=1)
    global_scale = float(np.quantile(residual, cfg.rho))
    weight = cfg.k / (cfg.k + cfg.kappa)
    return weight * local + (1.0 - weight) * global_scale + 1e-6


def _metrics(frame: pd.DataFrame, lower: np.ndarray, upper: np.ndarray) -> dict[str, float | int]:
    y = frame["label_value"].to_numpy(dtype=float)
    lower = np.clip(lower, 0.0, 1.0)
    upper = np.clip(upper, 0.0, 1.0)
    return {
        "n": int(len(frame)),
        "coverage": float(np.mean((lower <= y) & (y <= upper))),
        "sharpness": float(np.mean(upper - lower)),
    }


def evaluate_subset(cal: pd.DataFrame, test: pd.DataFrame, dataset: str, cfg: CareConfig) -> dict[str, object]:
    ref, holdout = _split_calibration(cal, cfg)
    hold_lower = _scales(ref, holdout, dataset, cfg, "lower")
    hold_upper = _scales(ref, holdout, dataset, cfg, "upper")
    score_error = holdout["score"].to_numpy(dtype=float) - holdout["label_value"].to_numpy(dtype=float)
    conformal_score = np.maximum(np.maximum(score_error, 0.0) / hold_lower, np.maximum(-score_error, 0.0) / hold_upper)
    multiplier = _q_high(conformal_score, cfg.target_coverage)
    test_lower_scale = _scales(ref, test, dataset, cfg, "lower")
    test_upper_scale = _scales(ref, test, dataset, cfg, "upper")
    center = test["score"].to_numpy(dtype=float)
    lower = center - multiplier * test_lower_scale
    upper = center + multiplier * test_upper_scale
    return {
        "metrics": _metrics(test, lower, upper),
        "multiplier": float(multiplier),
        "split": {
            "reference": int(len(ref)),
            "conformal": int(len(holdout)),
        },
    }


def evaluate_cell(score_root: str | Path, score_model: str, dataset: str, seed: int, cfg: CareConfig) -> dict[str, object]:
    splits = load_score_splits(score_root, score_model, dataset, seed)
    cal = {
        "positive": splits["cal_pos"],
        "negative": splits["cal_neg"],
        "combined": pd.concat([splits["cal_pos"], splits["cal_neg"]], ignore_index=True),
    }
    test = {
        "positive": splits["test_pos"],
        "negative": splits["test_neg"],
        "combined": splits["test_mix"],
    }
    return {
        "method": "CARE",
        "score_model": score_model,
        "dataset": dataset,
        "seed": int(seed),
        "config": cfg.__dict__,
        "subsets": {name: evaluate_subset(cal[name], test[name], dataset, cfg) for name in SUBSETS},
    }


def write_json(path: str | Path, data: object) -> Path:
    path = _rooted(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
