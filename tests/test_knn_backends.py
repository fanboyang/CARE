import numpy as np

from care.interval import PAPER_SEEDS, CareConfig, _knn, split_seed_for_score_seed


def test_cpu_backend_returns_exact_l2_neighbors():
    reference = np.asarray([[0.0, 0.0], [2.0, 0.0], [5.0, 0.0]], dtype=np.float32)
    target = np.asarray([[1.8, 0.0]], dtype=np.float32)

    indices = _knn(reference, target, 2, "cpu")

    assert indices.tolist() == [[1, 0]]


def test_config_accepts_explicit_backend():
    cfg = CareConfig(knn_backend="mps")

    assert cfg.knn_backend == "mps"


def test_paper_seed_policy_uses_score_seed_as_split_seed():
    assert PAPER_SEEDS == tuple(range(10))
    assert [split_seed_for_score_seed(seed) for seed in PAPER_SEEDS] == list(range(10))
