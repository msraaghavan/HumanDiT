import numpy as np
import torch

from humangen.eval.metrics import frechet_distance, knn_precision_recall, stats


def test_frechet_distance_closed_form():
    # FD between N(m1, a I) and N(m2, b I) in d dims = |m1-m2|^2 + d (sqrt(a) - sqrt(b))^2
    d = 8
    mu1, mu2 = np.zeros(d), np.full(d, 0.5)
    s1, s2 = np.eye(d) * 4.0, np.eye(d) * 1.0
    assert abs(frechet_distance(mu1, s1, mu2, s2) - (d * 0.25 + d * 1.0)) < 1e-6


def test_frechet_distance_same_distribution_is_small():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(4000, 16)), rng.normal(size=(4000, 16))
    assert frechet_distance(*stats(a), *stats(b)) < 0.1


def test_precision_recall_detects_mode_collapse():
    torch.manual_seed(0)
    real = torch.randn(2000, 8)
    collapsed = torch.randn(2000, 8) * 0.1  # all samples in a small region of the real support
    pr = knn_precision_recall(real, collapsed)
    assert pr["precision"] > 0.9 and pr["recall"] < 0.3
