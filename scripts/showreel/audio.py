"""Procedural soundtrack for the reel: every sound is synthesised here, and every cue comes from the page's own timeline
(window.CUES in reel.html), so each tick, whoosh and impact lands on the frame that causes it. No samples, no licences.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

SR = 48_000
RNG = np.random.default_rng(7)


def env_exp(n: int, decay: float) -> np.ndarray:
    return np.exp(-np.arange(n) / SR * decay)


def tone(f0, dur, harm=1, amp=1.0):
    t = np.arange(int(dur * SR)) / SR
    f = np.broadcast_to(np.asarray(f0, float), t.shape) if np.ndim(f0) else np.full(t.shape, float(f0))
    ph = 2 * np.pi * np.cumsum(f) / SR
    return amp * sum(np.sin(k * ph) / k**1.3 for k in range(1, harm + 1))


def onepole(x: np.ndarray, cutoff: np.ndarray | float, high: bool = False) -> np.ndarray:
    """Time-varying one-pole low-pass (or high-pass)."""
    c = np.broadcast_to(np.asarray(cutoff, float), x.shape)
    a = 1 - np.exp(-2 * np.pi * c / SR)
    y, s = np.empty_like(x), 0.0
    for i in range(len(x)):
        s += a[i] * (x[i] - s)
        y[i] = s
    return x - y if high else y


def bell(f, dur=1.6, idx=2.2):
    t = np.arange(int(dur * SR)) / SR
    mod = idx * np.exp(-t * 3.5) * np.sin(2 * np.pi * 3.5 * f * t)
    return np.sin(2 * np.pi * f * t + mod) * np.exp(-t * 2.4)


def kick(amp=1.0, dur=0.45):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 44 + 120 * np.exp(-t * 30)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7.5)
    click = onepole(RNG.standard_normal(n) * np.exp(-t * 900), 2500, high=True) * 0.35
    return amp * (body + click)


class Mix:
    def __init__(self, seconds: float):
        self.dry = np.zeros((int(seconds * SR) + SR, 2))
        self.wet = np.zeros_like(self.dry)

    def add(self, t: float, x: np.ndarray, pan: float = 0.0, gain: float = 1.0, send: float = 0.0) -> None:
        i = round(t * SR)
        if i < 0:
            x, i = x[-i:], 0
        x = x[: max(0, len(self.dry) - i)]
        lr = np.array([np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)]) * np.sqrt(2)
        seg = x[:, None] * lr[None, :] * gain
        self.dry[i : i + len(x)] += seg
        self.wet[i : i + len(x)] += seg * send


CHORDS = [  # (start, frequencies): D minor 9 -> Bb major 7 -> C add9 -> D minor 9
    (0.0, [73.42, 110.0, 146.83, 174.61, 220.0, 261.63, 329.63]),
    (8.0, [58.27, 87.31, 116.54, 146.83, 174.61, 220.0, 293.66]),
    (10.5, [65.41, 98.0, 130.81, 164.81, 196.0, 293.66, 329.63]),
    (13.0, [73.42, 110.0, 146.83, 174.61, 220.0, 261.63, 329.63]),
]


def pad(seconds: float) -> np.ndarray:
    n = int(seconds * SR)
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    for ci, (t0, freqs) in enumerate(CHORDS):
        t1 = CHORDS[ci + 1][0] if ci + 1 < len(CHORDS) else seconds + 1
        w = np.clip((t - t0 + 0.15) / 0.3, 0, 1) * np.clip((t1 - t + 0.15) / 0.3, 0, 1)
        for k, f in enumerate(freqs):
            for ch, det in enumerate((-0.12, 0.12)):
                ph = 2 * np.pi * (f + det * (k + 1)) * t + k * 1.3
                v = sum(np.sin(h * ph) / h**1.6 for h in (1, 2, 3))
                lfo = 0.75 + 0.25 * np.sin(2 * np.pi * (0.2 * 18 / 15) * t + k)  # periodic over the loop
                out[:, ch] += v * lfo * w / (len(freqs) * (1 + 0.35 * k))
    return out * 0.32


def synth(cues: list[dict], seconds: float, path: Path) -> None:
    m = Mix(seconds)
    for c in cues:
        t, k = c["t"], c["kind"]
        if k == "pulse":  # a soft kick on every beat (120 bpm), from the first cut to the end card
            for beat in np.arange(t, c["t1"], 0.5):
                m.add(beat, kick(0.35, 0.3), gain=0.55)
        elif k == "shutter":  # the scene-5 wipe: a fast down-then-up whoosh with a low thud in the middle
            n = int(c["dur"] * SR)
            u = np.linspace(0, 1, n)
            sweep = (
                onepole(RNG.standard_normal(n), 400 + 6000 * np.abs(np.sin(np.pi * u)))
                * np.sin(np.pi * u) ** 1.5
            )
            m.add(t, sweep, gain=0.5, send=0.3)
            m.add(t + c["dur"] * 0.45, kick(0.6, 0.35), gain=0.6)
        elif k == "impact":
            n = int(1.6 * SR)
            boom = tone(38, 1.6) * env_exp(n, 2.6)
            noise = onepole(RNG.standard_normal(n), 900) * env_exp(n, 5)
            m.add(t, kick(1.0), gain=0.9)
            m.add(t, boom + noise * 0.8, gain=0.55, send=0.5)
        elif k == "title":
            m.add(t, tone(55, 1.4) * env_exp(int(1.4 * SR), 2.2), gain=0.5, send=0.3)
            m.add(t + 0.05, bell(587.33, 2.0), gain=0.12, send=0.8)
        elif k == "riser":
            n = int(c["dur"] * SR)
            u = np.linspace(0, 1, n)
            nz = onepole(RNG.standard_normal(n), 300 + 7000 * u**2)
            nz = onepole(nz, 150 + 900 * u, high=True)
            sw = tone(180 + 520 * u**2, c["dur"]) * 0.25
            m.add(t, (nz + sw) * u**2.2, gain=0.5, send=0.4)
        elif k == "tick":
            n = int(0.03 * SR)
            f = 1500 + 1300 * c["v"]
            m.add(t, np.sin(2 * np.pi * f * np.arange(n) / SR) * env_exp(n, 160), pan=0.35, gain=0.22)
        elif k == "swipe":
            n = int(0.34 * SR)
            u = np.linspace(0, 1, n)
            whoosh = onepole(RNG.standard_normal(n), 5200 * (1 - u) ** 1.5 + 300) * np.sin(np.pi * u) ** 2
            m.add(t - 0.04, whoosh, pan=-0.3 + 0.2 * c["v"], gain=0.45)
            note = [587.33, 698.46, 880.0, 1046.5][c["v"] % 4]
            m.add(t + 0.02, bell(note, 1.2, 1.8), pan=0.2, gain=0.14, send=0.7)
        elif k == "pop":
            n = int(0.02 * SR)
            f = 700 + 900 * c["v"]
            m.add(
                t,
                np.sin(2 * np.pi * f * np.arange(n) / SR) * env_exp(n, 220),
                pan=c["pan"],
                gain=0.07,
                send=0.3,
            )
        elif k == "race":
            u = np.linspace(0, 1, int(c["dur"] * SR))
            m.add(t, tone(110 * 2 ** (2 * u), c["dur"], harm=4) * u * 0.5, gain=0.25, send=0.2)
        elif k == "chime":
            for f, g in ((880.0, 0.16), (1318.5, 0.1)):
                m.add(t, bell(f, 1.8), gain=g, send=0.8)
        elif k == "fall":
            n = int(c["dur"] * SR)
            u = np.linspace(0, 1, n)
            f = 1300 * (1 - u) ** 2 + 180
            trem = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * np.cumsum(38 * (1 - u) + 6) / SR))
            m.add(t, tone(f, c["dur"]) * trem * (1 - u) ** 0.5, gain=0.12, send=0.2)
        elif k == "blip":
            n = int(0.06 * SR)
            f = 440 * 2 ** ((3 + 2 * c["v"]) / 12)
            sq = np.sign(np.sin(2 * np.pi * f * np.arange(n) / SR))
            m.add(t, onepole(sq, 3000) * env_exp(n, 45), pan=-0.4 + 0.27 * c["v"], gain=0.1, send=0.3)
        elif k == "bell":
            for f, g, p in (
                (293.66, 0.2, -0.3),
                (440.0, 0.16, 0.3),
                (698.46, 0.12, 0.0),
                (1174.66, 0.07, 0.1),
            ):
                m.add(t, bell(f, 2.6, 1.6), pan=p, gain=g, send=0.9)
        elif k == "reverse":
            n = int(c["dur"] * SR)
            u = np.linspace(0, 1, n)
            m.add(t, onepole(RNG.standard_normal(n), 2500, high=True) * u**3, gain=0.35, send=0.3)
    n = int(seconds * SR)
    ir_n = int(1.8 * SR)
    ir = RNG.standard_normal((ir_n, 2)) * env_exp(ir_n, 3.2)[:, None]
    ir[:, 0], ir[:, 1] = onepole(ir[:, 0], 5000), onepole(ir[:, 1], 5000)
    wet = np.stack([fftconvolve(m.wet[:, ch], ir[:, ch])[: len(m.wet)] for ch in (0, 1)], axis=1) * 0.06
    out = m.dry + wet
    out[:n] += pad(seconds)
    # the loop point: the reverb tail past 15 s wraps around to the start
    out[: len(out) - n] += out[n:]
    out = np.tanh(out[:n] * 1.1)
    out /= np.abs(out).max() / 0.89
    fade = int(0.004 * SR)
    out[:fade] *= np.linspace(0, 1, fade)[:, None]
    out[-fade:] *= np.linspace(1, 0, fade)[:, None]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((out * 32767).astype("<i2").tobytes())
