import numpy as np
import torch

from humangen.data.attributes import (
    INVERT,
    Probe,
    calibrate_abstention,
    evaluate,
    fit_linear_probe,
    select_attributes,
)


def _synthetic(n=3000, d=32, k=3, seed=0):
    w = np.random.default_rng(123).normal(size=(k, d))  # same ground truth for every split
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float32)
    y = (x @ w.T + rng.normal(scale=0.5, size=(n, k)) > 0.8).astype(bool)
    return x, y


def test_probe_learns_and_abstains():
    x, y = _synthetic()
    xv, yv = _synthetic(seed=1)
    w, b, mu, sig = fit_linear_probe(x, y, xv, yv, epochs=10)
    names = ["Smiling", "Eyeglasses", "No_Beard"]
    probe = Probe(names, w, b, mu, sig, np.zeros(3), np.ones(3))
    m = evaluate(probe, xv, yv)
    assert all(v["balanced_acc"] > 0.85 for v in m.values())
    p = torch.sigmoid(probe.logits(torch.from_numpy(xv))).numpy()
    lo, hi = calibrate_abstention(p, yv, 0.9, 0.97)
    probe.lo, probe.hi = lo, hi
    tern = probe.ternary(torch.from_numpy(xv))
    assert set(np.unique(tern)) <= {0, 1, 2}
    # "present" labels must meet the precision target (on the calibration split); inverted attr flips
    for j, n in enumerate(names):
        truth = ~yv[:, j] if n in INVERT else yv[:, j]
        if (tern[:, j] == 2).sum() > 20:
            assert truth[tern[:, j] == 2].mean() >= 0.88
    assert probe.display_names[2] == "Beard"


def test_selection_excludes_subjective():
    m = {"Attractive": {"balanced_acc": 0.99}, "Eyeglasses": {"balanced_acc": 0.97}, "Smiling": {"balanced_acc": 0.8}}
    assert select_attributes(m, 0.88, 12) == ["Eyeglasses"]


def test_calibration_reaches_rare_attributes():
    """Rare attribute whose 90%-precision region lies above p = 0.99 (class-balanced probes do this)."""
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.uniform(0.0, 0.995, 10000), rng.uniform(0.995, 1.0, 200)])[:, None]
    y = np.concatenate([np.zeros(10000, bool), np.ones(200, bool)])[:, None]
    lo, hi = calibrate_abstention(p, y, 0.9, 0.97)
    present = p[:, 0] >= hi[0]
    assert present.sum() >= 150 and y[present, 0].mean() >= 0.9
    assert (p[:, 0] <= lo[0]).sum() > 9000  # confident "absent" region still found
