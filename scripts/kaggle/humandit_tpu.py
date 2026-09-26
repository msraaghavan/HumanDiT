"""HumanDiT follow-up runs on a Kaggle TPU v5e-8 (torch_xla). Pushed as a Kaggle script kernel.

Inputs (private Kaggle datasets): humandit-code (this repo at a commit, as humandit-code.tar.gz) and humandit-weights
(teacher EMA, attribute probe + report, real FFHQ DINOv2 statistics). Internet is used for FFHQ-256 and the RAEv2 /
DINOv2 / clean-fid weights.

Stage 1 rehearses every job at toy size; stage 2 (the real run) starts only if all of stage 1 passed, so a porting
bug costs minutes of TPU quota, not hours. Jobs:
    encode  FFHQ-256 -> RAEv2 latents (70k x [image, h-flip]), full-precision matmuls
    label   pseudo-label with the existing probe; coverage must match the original (reproducibility check)
    gan     GAN student with the fixed discriminator noise levels + teacher anchors, data-parallel on 8 chips,
            same step count and batch as the original run (2053 x 64)
    eval    that student: FID-50k, FD_DINOv2, precision/recall, memorisation
    control teacher at the tuned guidance 1.55: attribute control + base rates
    steps   teacher at IG 1.55: FID-10k at 10 and 25 steps (50 steps was measured in the guidance sweep)
    stats   latent statistics at IG 1.00 / 1.55 / 1.78
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

T0 = time.time()
BUDGET_H = 8.3  # Kaggle TPU sessions stop at 9 h; leave time for outputs to be saved
OUT = Path("/kaggle/working")
INP = Path("/kaggle/input")


def big_disk() -> Path:
    cands = [Path(p) for p in ("/kaggle/temp", "/tmp", "/root", str(OUT)) if Path(p).exists()]
    best = max(cands, key=lambda p: shutil.disk_usage(p).free)
    print(f"[driver] work disk {best}: {shutil.disk_usage(best).free / 1e9:.0f} GB free", flush=True)
    return best


WORK = big_disk() / "humandit"
REPO = WORK / "repo"
ENV = {
    **os.environ,
    "PJRT_DEVICE": "TPU",
    "PYTHONPATH": str(REPO / "src"),
    "PYTHONUNBUFFERED": "1",
    "HUMANGEN_PRETRAINED": str(WORK / "pretrained" / "raev2"),
    "HF_HOME": str(WORK / "hf"),
    "TORCH_HOME": str(WORK / "torch"),
}


def sh(cmd: str, log: Path | None = None, check: bool = True) -> int:
    print(f"\n$ {cmd}", flush=True)
    t = time.time()
    with open(log, "a") if log else open(os.devnull, "w") as lf:
        p = subprocess.Popen(
            cmd,
            shell=True,
            cwd=REPO if REPO.exists() else None,
            env=ENV,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in p.stdout:
            lf.write(line)
            if (
                not line.startswith(("W0", "I0")) or "rror" in line
            ):  # skip absl INFO/WARN spam in the notebook log
                print(line, end="", flush=True)
        rc = p.wait()
    print(f"[driver] exit {rc} after {(time.time() - t) / 60:.1f} min", flush=True)
    if check and rc != 0:
        raise RuntimeError(f"command failed ({rc}): {cmd}")
    return rc


def setup() -> None:
    code = glob.glob(str(INP / "**" / "humandit-code.tar.gz"), recursive=True)
    weights = glob.glob(str(INP / "**" / "teacher_ema.safetensors"), recursive=True)
    assert code and weights, f"missing inputs: code={code} weights={weights}"
    wdir = Path(weights[0]).parent
    REPO.mkdir(parents=True, exist_ok=True)
    with tarfile.open(code[0]) as t:
        t.extractall(REPO, filter="data")
    for rel, src in {
        "runs/teacher_b/ckpt/ema.safetensors": wdir / "teacher_ema.safetensors",
        "runs/attributes/attribute_probe.pt": wdir / "attribute_probe.pt",
        "runs/attributes/attributes.json": wdir / "attributes.json",
        "runs/eval/ffhq256_dinov2_real.npz": wdir / "ffhq256_dinov2_real.npz",
        "runs/eval_toy/ffhq256_dinov2_real.npz": wdir / "ffhq256_dinov2_real.npz",
    }.items():
        (REPO / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, REPO / rel)
    sh(
        "pip install -q clean-fid omegaconf safetensors huggingface_hub pyarrow scikit-learn imageio 2>&1 | tail -2",
        check=False,
    )
    sh(
        "python -c 'import sys, torch, torch_xla; print(sys.version); print(torch.__version__, torch_xla.__version__)'"
    )


def jobs(toy: bool) -> list[tuple[str, str, list[str]]]:
    """(name, command, output paths to keep). Toy sizes exercise every code path in minutes."""
    d, c = WORK / ("data_toy" if toy else "data"), WORK / ("cache_toy" if toy else "cache")
    ev, ex = ("runs/eval_toy", "runs/eval_extra_toy") if toy else ("runs/eval", "runs/eval_extra")
    gan = "runs/student_gan_fixed_toy" if toy else "runs/student_gan_fixed"
    teacher, probe = "runs/teacher_b/ckpt/ema.safetensors", "runs/attributes/attribute_probe.pt"
    common = f"--data {d}/ffhq-256 --cache {c}/ffhq256 --probe {probe}"
    n = lambda full, small: small if toy else full  # noqa: E731
    return [
        (
            "encode",
            f"python scripts/prepare_data.py ffhq --data-dir {d} --cache-dir {c} --batch 64 --workers 8"
            + (" --shards 1 --limit 256" if toy else ""),
            [],
        ),
        (
            "label",
            f"python scripts/label_ffhq.py --cache {c}/ffhq256" + (" --tol 1.0" if toy else ""),
            [f"{c}/ffhq256/attributes.json"],
        ),
        (
            "gan",
            f"python scripts/onestep.py gan --tpu --teacher {teacher} --cache {c}/ffhq256 --out {gan} "
            f"--gan-batch 64 --anchor-batch 32 --anchor-pairs {n(6144, 256)} --anchor-steps {n(25, 2)} "
            f"--max-steps {n(2053, 3)} --minutes 300 --sample-every {n(500, 1000)} --decode",
            [gan],
        ),
        (
            "eval",
            f"python scripts/evaluate.py --ckpt {gan}/student.safetensors --kind student --steps 1 "
            f"--tag gan_fixed_1step --n {n(50000, 250)} --metrics fid,fd_dinov2,pr,memo {common} --out {ev}",
            [ev],
        ),
        (
            "control",
            f"python scripts/evaluate.py --ckpt {teacher} --tag teacher_tuned_control --ig-scale 1.55 "
            f"--steps {n(50, 4)} --n {n(10000, 128)} --control-n {n(256, 64)} --metrics control,memo {common} "
            f"--out {ev}",
            [ev],
        ),
        *[
            (
                f"steps{s}",
                f"python scripts/evaluate.py --ckpt {teacher} --tag teacher_tuned_{s}step_10k "
                f"--ig-scale 1.55 --steps {n(s, 2)} --n {n(10000, 250)} --metrics fid {common} --out {ex}",
                [ex],
            )
            for s in (10, 25)
        ],
        (
            "stats",
            f"python scripts/latent_stats.py --ckpt {teacher} --cache {c}/ffhq256 --ig 1.0 1.55 1.78 "
            f"--out {ev}/latent_stats_tuned.json" + (" --n 16 --steps 2" if toy else " --n 256 --batch 32"),
            [ev],
        ),
    ]


def run_stage(toy: bool) -> dict[str, str]:
    status, dead = {}, set()
    needs = {
        "label": {"encode"},
        "gan": {"encode", "label"},
        "eval": {"gan"},
        "control": {"encode"},
        "stats": {"encode"},
    }
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)
    for name, cmd, keep in jobs(toy):
        tag = f"{'toy_' if toy else ''}{name}"
        if (time.time() - T0) / 3600 > BUDGET_H:
            status[tag] = "skipped (session time budget)"
            continue
        if needs.get(name, set()) & dead:
            status[tag] = "skipped (dependency failed)"
            dead.add(name)
            continue
        rc = sh(cmd, logs / f"{tag}.log", check=False)
        status[tag] = "ok" if rc == 0 else f"FAILED ({rc})"
        if rc != 0:
            dead.add(name)
        for rel in keep:  # copy results out as soon as each job finishes (a later crash cannot lose them)
            src = REPO / rel if not rel.startswith("/") else Path(rel)
            if src.exists():
                dst = OUT / "results" / (src.relative_to(REPO) if src.is_relative_to(REPO) else src.name)
                if src.is_dir():
                    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*_images"))
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(src, dst)
        (OUT / "status.json").write_text(json.dumps(status, indent=2))
    return status


def main() -> None:
    setup()
    toy = run_stage(toy=True)
    print(json.dumps(toy, indent=2), flush=True)
    if any(v != "ok" for v in toy.values()) or "--toy-only" in sys.argv:
        print("[driver] rehearsal did not fully pass (or --toy-only); not starting the real run", flush=True)
        return
    shutil.rmtree(OUT / "results", ignore_errors=True)  # keep only real results
    real = run_stage(toy=False)
    print(json.dumps({**toy, **real}, indent=2), flush=True)


if __name__ == "__main__":
    main()
