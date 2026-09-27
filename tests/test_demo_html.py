"""The demo's generated HTML: what the motion layer animates must say the same thing with no script at all."""

import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("gradio")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import gradio_app as ga


def test_race_marks_the_small_sample_and_the_best_gap():
    out = ga.race_html(
        [("Instant", 0.08), ("25 steps", 1.9)],
        [("Instant", 20.0, 50_000), ("10 steps", 5.9, 10_000), ("25 steps", 3.4, 50_000)],
        "a test GPU",
    )
    assert out.count('class="hd-lane time"') == 2 and out.count('class="hd-lane gapl"') == 3
    assert 'data-suffix="*">5.9*' in out and "10,000 faces" in out  # the FID-10k value is labelled as such
    assert re.search(r'hd-track gap best"><i[^>]*></i></div><b data-suffix="">3.4<', out)
    # without the script, bars already sit at their final length: time relative to the slowest run
    assert "scaleX(1.0000)" in out and f"scaleX({0.08 / 1.9:.4f})" in out


def test_scoreboard_is_log_scaled_and_ordered_by_the_measurements():
    evals = {
        k: {"fid_clean": f}
        for k, f in (("legacy_dcgan_64px", 177.6), ("gan_fixed_1step", 20.0), ("teacher_tuned_25step", 3.41))
    }
    out = ga.scoreboard_html(evals)
    widths = [float(w) for w in re.findall(r'data-w="([\d.]+)"', out)]
    assert len(widths) == 3 and widths[0] > widths[1] > widths[2] > 0.2  # log scale keeps 3.4 visible
    assert 'class="hd-srow best"' in out and "<em>Best quality</em>" in out and "<em>Instant</em>" in out
    assert ga.scoreboard_html({}) == ""


def test_file_urls_are_relative_to_the_page():
    url = ga.file_url(ga.ASSETS / "web" / "motion.json")
    assert url.startswith("gradio_api/file=") and url.endswith("web/motion.json") and " " not in url
