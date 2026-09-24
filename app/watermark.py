"""Invisible watermark + provenance metadata for every generated image.

Uses the DWT-DCT watermark from `invisible-watermark` (as used by Stable Diffusion releases). It survives
PNG round-trips and mild re-encoding, not deliberate removal: it is a provenance aid, not a guarantee.
"""

from __future__ import annotations

import numpy as np
from imwatermark import WatermarkDecoder, WatermarkEncoder
from PIL import Image, PngImagePlugin

MARK = b"HumanDiT"  # 8 bytes = 64 bits


def add_watermark(img: Image.Image) -> Image.Image:
    enc = WatermarkEncoder()
    enc.set_watermark("bytes", MARK)
    bgr = np.asarray(img.convert("RGB"))[:, :, ::-1].copy()
    out = enc.encode(bgr, "dwtDct")
    return Image.fromarray(out[:, :, ::-1])


def check_watermark(img: Image.Image) -> bool:
    dec = WatermarkDecoder("bytes", len(MARK) * 8)
    bgr = np.asarray(img.convert("RGB"))[:, :, ::-1].copy()
    return dec.decode(bgr, "dwtDct") == MARK


def provenance_pnginfo(details: str) -> PngImagePlugin.PngInfo:
    info = PngImagePlugin.PngInfo()
    info.add_text("AI-generated", "true")
    info.add_text("Generator", "HumanDiT (RAEv2 + DiT^DH), research demo; synthetic person, not a real individual")
    info.add_text("Details", details)
    return info
