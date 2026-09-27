"""Invisible watermark + provenance metadata for every generated image.

Uses Adobe's TrustMark (MIT, the neural watermark used in the Content Credentials / C2PA ecosystem) with a
BCH-protected text payload. Measured on generated faces (scripts/watermark_eval.py): recovered from 100% of images
after PNG, JPEG q75 and 2x down/up-scaling, at 43 dB PSNR. It is a provenance aid, not a guarantee: TrustMark ships
a remover, so a determined adversary can strip it.

The earlier DWT-DCT watermark (invisible-watermark 0.2) recovered 0 of 6 marks on 256x256 faces here, even straight
after embedding, and changed pixels by up to 179 levels. It was replaced.
"""

from __future__ import annotations

from functools import lru_cache

from PIL import Image, PngImagePlugin

MARK = "HumanDiT"


@lru_cache(maxsize=1)
def _tm(device: str = "cpu"):
    from trustmark import TrustMark

    return TrustMark(verbose=False, model_type="Q", device=device, loadRemover=False)


def load_watermarker(device: str = "cpu") -> None:
    """Load (and on first use download) TrustMark's weights now rather than inside the first job."""
    _tm(device)


def add_watermark(img: Image.Image, device: str = "cpu") -> Image.Image:
    return _tm(device).encode(img.convert("RGB"), MARK, MODE="text")


def check_watermark(img: Image.Image, device: str = "cpu") -> bool:
    """True only if the decoded payload is exactly MARK (the model's own 'present' flag alone false-fires)."""
    secret, present, _ = _tm(device).decode(img.convert("RGB"), MODE="text")
    return bool(present) and secret == MARK


def provenance_pnginfo(details: str) -> PngImagePlugin.PngInfo:
    info = PngImagePlugin.PngInfo()
    info.add_text("AI-generated", "true")
    info.add_text(
        "Generator", "HumanDiT (RAEv2 + DiT^DH), research demo; synthetic person, not a real individual"
    )
    info.add_text("Details", details)
    return info
