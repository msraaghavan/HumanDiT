"""Hugging Face Space entry point for the HumanDiT demo (ZeroGPU hardware).

scripts/build_space.py copies this file to the root of the Space, next to app/, src/ and docs/assets/. At start-up it
downloads the trained weights and measured results from the model repo (HUMANDIT_MODEL_REPO) and the RAEv2 decoder
from its authors' repo, then starts app/gradio_app.py. On ZeroGPU a GPU is attached only while a job runs.
"""

import os
import sys
from pathlib import Path

import spaces  # noqa: F401  (must be imported before torch initialises CUDA)

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "app")]
# ZeroGPU runs on Ampere-or-newer GPUs: bf16, and no capability query while no GPU is attached
os.environ.setdefault("HUMANGEN_PRECISION", "bf16")
os.environ.setdefault("HUMANGEN_PRETRAINED", str(ROOT / "pretrained" / "raev2"))

import gradio_app  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

from humangen.rae import ensure_weights  # noqa: E402

MODEL_REPO = os.environ.get("HUMANDIT_MODEL_REPO", "__MODEL_REPO__")

# HUMANDIT_MODEL_DIR: a local copy of the model repo (dist/model), to test the Space before uploading anything
local = os.environ.get("HUMANDIT_MODEL_DIR")
weights = Path(local) if local else Path(snapshot_download(MODEL_REPO))
ensure_weights(Path(os.environ["HUMANGEN_PRETRAINED"]), encoder=False)  # decoder + latent statistics only
gradio_app.main(
    [
        "--teacher",
        str(weights / "teacher_b" / "ema.safetensors"),
        "--student",
        str(weights / "student_gan_fixed" / "student.safetensors"),
        "--probe",
        str(weights / "attributes" / "attribute_probe.pt"),
        "--eval-dirs",
        str(weights / "eval"),
        "--no-warm-up",
    ]
)
