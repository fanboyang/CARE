import numpy as np

from care.interval import CareConfig, _knn


def test_cpu_backend_returns_exact_l2_neighbors():
    reference = np.asarray([[0.0, 0.0], [2.0, 0.0], [5.0, 0.0]], dtype=np.float32)
    target = np.asarray([[1.8, 0.0]], dtype=np.float32)

    indices = _knn(reference, target, 2, "cpu")

    assert indices.tolist() == [[1, 0]]


def test_config_accepts_explicit_backend():
    cfg = CareConfig(knn_backend="mps")

    assert cfg.knn_backend == "mps"
