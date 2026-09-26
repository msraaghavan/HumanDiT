"""Watermark round trip on a generated face (skipped when the demo extra / TrustMark models are unavailable)."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("trustmark")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))


@pytest.mark.slow
def test_trustmark_roundtrip_and_no_false_positive():
    from watermark import add_watermark, check_watermark

    grid = Path(__file__).resolve().parents[1] / "docs" / "assets" / "teacher_final_grid.jpg"
    img = Image.open(grid).convert("RGB").crop((2, 2, 258, 258))  # one generated face
    marked = add_watermark(img)
    assert check_watermark(marked)
    assert not check_watermark(img)
    diff = np.asarray(marked, float) - np.asarray(img, float)
    assert 10 * np.log10(255**2 / np.mean(diff**2)) > 38  # invisible (real faces measure 40-47 dB)
