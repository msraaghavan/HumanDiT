"""HumanDiT follow-up runs on a cloud VM: a Colab GPU (how the published follow-ups ran: A100) or TPU runtime driven by
the Colab CLI (`colab exec -f scripts/colab/driver.py`, see scripts/colab/README.md), or a Kaggle script kernel.

TPU (torch_xla) is UNVERIFIED: its first real run (Colab v6e-1) hit two known bugs that were deliberately left unfixed
-- see docs/compute_platforms.md section 3. Use a CUDA GPU.

Inputs: humandit-code.tar.gz (this repo at a commit) plus the weights (teacher EMA, attribute probe + report, real FFHQ
DINOv2 statistics, fixed images for the platform check). On Colab they are uploaded to /content/inputs
(scripts/colab/upload_inputs.sh); on Kaggle they are private datasets under /kaggle/input. Internet is used for
FFHQ-256 and the RAEv2 / DINOv2 / clean-fid weights.

HUMANDIT_STAGES picks the stages (default "toy,real"): toy rehearses every job of the main stage at toy size and stops
the driver if any job fails, so a porting bug costs minutes, not hours; check runs scripts/platform_check.py
(bake-off); real is the first follow-up run and fix the second (below); "pack" alone re-packs earlier outputs.
Results are packed into 40 MB parts under /content/dl. Jobs of the real stage:
    encode  FFHQ-256 -> RAEv2 latents (70k x [image, h-flip]), full-precision matmuls
    label   pseudo-label with the existing probe; coverage must match the original (reproducibility check)
    gan     GAN student with the fixed discriminator noise levels + teacher anchors (data-parallel over TPU chips with
            --tpu), same step count and batch as the original run (2053 x 64)
    eval    that student: FID-50k, FD_DINOv2, precision/recall, memorisation
    control teacher at the tuned guidance 1.55: attribute control + base rates
    steps   teacher at IG 1.55: FID-10k at 10 and 25 steps (50 steps was measured in the guidance sweep)
    stats   latent statistics at IG 1.00 / 1.55 / 1.78
Jobs of the fix stage (the fixed GAN lost facial hair; scripts/control_diagnosis.py):
    encode, label   as above
    warm            download the clean-fid reference statistics and Inception once (two jobs share the GPU)
    teacher25       teacher, IG 1.55, 25 steps: FID-50k, FD_DINOv2, precision/recall (in the background)
    gan_attrd       GAN with an attribute-aware discriminator (attribute-token heads + matching-aware term), same
                    data, anchors, batch and steps as the fixed GAN, logging control rates every 125 steps
    gan_anchoronly  ablation: the same run with the adversarial term off (anchors only)
    eval_*          each student: FID-50k, FD_DINOv2, precision/recall, control, memorisation
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

T0 = time.time()
KAGGLE = "COLAB_RELEASE_TAG" not in os.environ and Path("/kaggle/input").exists()  # Colab also has /kaggle/*
GPU = shutil.which("nvidia-smi") is not None  # the same jobs also run on a CUDA GPU (no --tpu, no PJRT)
# which stages to run: toy (rehearsal), check (platform agreement + speed, scripts/platform_check.py), real or fix
STAGES = os.environ.get("HUMANDIT_STAGES", "toy,real").split(",")
MAIN = "fix" if "fix" in STAGES else "real"  # the stage the toy rehearsal mirrors
BACKGROUND = {"teacher25"}  # runs alongside the GAN jobs on the same GPU (an A100-40GB holds both)
BUDGET_H = 8.3  # Kaggle stops TPU sessions at 9 h (Colab: a local deadman); leave time to save outputs
OUT = Path("/kaggle/working") if KAGGLE else Path("/content/humandit_out")
INP = Path("/kaggle/input") if KAGGLE else Path("/content/inputs")
OUT.mkdir(parents=True, exist_ok=True)


def big_disk() -> Path:
    cands = [Path(p) for p in ("/kaggle/temp", "/content", "/tmp", "/root", str(OUT)) if Path(p).exists()]
    best = max(cands, key=lambda p: shutil.disk_usage(p).free)
    print(f"[driver] work disk {best}: {shutil.disk_usage(best).free / 1e9:.0f} GB free", flush=True)
    return best


WORK = big_disk() / "humandit"
REPO = WORK / "repo"
ENV = {
    **os.environ,
    **({} if GPU else {"PJRT_DEVICE": "TPU"}),
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
            encoding="utf-8",
            errors="replace",
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
    # pin the image's torch / torchvision / torch_xla: an unpinned dependency upgrade would break torch_xla
    sh(
        "pip freeze | grep -iE '^(torch|torchvision|torch[-_]xla)==' > /tmp/pins.txt; cat /tmp/pins.txt; "
        "pip install -q -c /tmp/pins.txt clean-fid omegaconf safetensors huggingface_hub pyarrow scikit-learn "
        "imageio tensorboard 2>&1 | tail -2",
        check=False,
    )
    sh("python -c 'import sys, torch; print(sys.version); print(torch.__version__, torch.version.cuda)'")
    if not GPU:
        sh("python -c 'import torch_xla; print(torch_xla.__version__)'")


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
            f"python scripts/onestep.py gan{'' if GPU else ' --tpu'} --teacher {teacher} --cache {c}/ffhq256 --out {gan} "
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


def fix_jobs(toy: bool) -> list[tuple[str, str, list[str]]]:
    """The facial-hair follow-up (see the module docstring); encode and label as in the real stage."""
    base = {name: (cmd, keep) for name, cmd, keep in jobs(toy)}
    c = WORK / ("cache_toy" if toy else "cache")
    ev = "runs/eval_toy" if toy else "runs/eval"
    teacher, probe = "runs/teacher_b/ckpt/ema.safetensors", "runs/attributes/attribute_probe.pt"
    d = WORK / ("data_toy" if toy else "data")
    common = f"--data {d}/ffhq-256 --cache {c}/ffhq256 --probe {probe}"
    n = lambda full, small: small if toy else full  # noqa: E731
    out = [
        ("encode", *base["encode"]),
        ("label", *base["label"]),
        (
            "warm",
            "python -c 'import torch; from humangen.eval.metrics import cleanfid; fid, _ = cleanfid(); "
            '[fid.get_reference_statistics("ffhq", 256, mode=m, split="trainval70k") '
            'for m in ("clean", "legacy_tensorflow")]; '
            'fid.build_feature_extractor("clean", torch.device("cuda"))\''
            if GPU
            else "true",
            [],
        ),
        (
            "teacher25",
            f"python scripts/evaluate.py --ckpt {teacher} --tag teacher_tuned_25step --ig-scale 1.55 "
            f"--steps {n(25, 2)} --n {n(50000, 250)} --metrics fid,fd_dinov2,pr {common} --out {ev}",
            [ev],
        ),
    ]
    for name, flags in (("attrd", "--d-attr-heads --d-mismatch 1"), ("anchoronly", "--adv-weight 0")):
        run = f"runs/student_gan_{name}{'_toy' if toy else ''}"
        out.append(
            (
                f"gan_{name}",
                f"python scripts/onestep.py gan{'' if GPU else ' --tpu'} --teacher {teacher} --cache {c}/ffhq256 "
                f"--out {run} --gan-batch 64 --anchor-batch 32 --anchor-pairs {n(6144, 256)} "
                f"--anchor-steps {n(25, 2)} --max-steps {n(2053, 3)} --minutes 300 --sample-every {n(500, 1000)} "
                f"--control-every {n(125, 2)} --probe {probe} --decode {flags}",
                [run],
            )
        )
        out.append(
            (
                f"eval_{name}",
                f"python scripts/evaluate.py --ckpt {run}/student.safetensors --kind student --steps 1 "
                f"--tag gan_{name}_1step --n {n(50000, 250)} --control-n {n(256, 16)} "
                f"--metrics fid,fd_dinov2,pr,control,memo {common} --out {ev}",
                [ev],
            )
        )
    return out


STATUS: dict[
    str, str
] = {}  # every job of every stage: "running" / "ok (x.x min)" / "FAILED (rc) ..." / "skipped ..."


def save_status() -> None:
    (OUT / "status.json").write_text(json.dumps(STATUS, indent=2))


def keep_results(keep: list[str]) -> None:
    """Copy a job's outputs to OUT/results as soon as it finishes (a later crash cannot lose them)."""
    for rel in keep:
        src = REPO / rel if not rel.startswith("/") else Path(rel)
        if src.exists():
            dst = OUT / "results" / (src.relative_to(REPO) if src.is_relative_to(REPO) else src.name)
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*_images"))
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dst)


def run_stage(toy: bool, which: str = "real") -> dict[str, str]:
    status, dead = {}, set()
    needs = {
        "label": {"encode"},
        "gan": {"encode", "label"},
        "eval": {"gan"},
        "control": {"encode"},
        "stats": {"encode"},
        "gan_attrd": {"encode", "label"},
        "eval_attrd": {"gan_attrd"},
        "gan_anchoronly": {"encode", "label"},
        "eval_anchoronly": {"gan_anchoronly"},
    }
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)
    bg: list = []  # background jobs: (tag, name, process, start time, keep, log file)

    def reap(block: bool) -> None:
        for item in list(bg):
            tag, name, proc, t, keep, lf = item
            rc = proc.wait() if block else proc.poll()
            if rc is None:
                continue
            lf.close()
            status[tag] = "ok" if rc == 0 else f"FAILED ({rc})"
            STATUS[tag] = f"{status[tag]} ({(time.time() - t) / 60:.1f} min)"
            if rc != 0:
                dead.add(name)
            keep_results(keep)
            bg.remove(item)
            save_status()
            print(f"[driver] background job {tag}: {STATUS[tag]}", flush=True)

    for name, cmd, keep in fix_jobs(toy) if which == "fix" else jobs(toy):
        reap(block=False)
        tag = f"{'toy_' if toy else ''}{name}"
        if (time.time() - T0) / 3600 > BUDGET_H:
            status[tag] = "skipped (session time budget)"
            continue
        if needs.get(name, set()) & dead:
            status[tag] = "skipped (dependency failed)"
            dead.add(name)
            continue
        STATUS[tag] = "running"
        save_status()
        t = time.time()
        if name in BACKGROUND:
            print(f"\n$ {cmd}   [background]", flush=True)
            lf = open(logs / f"{tag}.log", "a")  # noqa: SIM115
            proc = subprocess.Popen(cmd, shell=True, cwd=REPO, env=ENV, stdout=lf, stderr=subprocess.STDOUT)
            bg.append((tag, name, proc, t, keep, lf))
            continue
        rc = sh(cmd, logs / f"{tag}.log", check=False)
        status[tag] = "ok" if rc == 0 else f"FAILED ({rc})"
        STATUS[tag] = f"{status[tag]} ({(time.time() - t) / 60:.1f} min)"
        if rc != 0:
            dead.add(name)
        keep_results(keep)
        save_status()
    reap(block=True)
    for tag, s in status.items():
        STATUS.setdefault(tag, s)  # skipped jobs
    save_status()
    return status


def run_check() -> bool:
    """Platform agreement + speed (scripts/platform_check.py): the default precision policy with timings, and a
    full-precision reference. Outputs go to OUT/check/{default,fp32}."""
    images = glob.glob(str(INP / "**" / "fixed_images.npy"), recursive=True)
    ok = bool(images)
    for mode, flags in (("default", "--timing"), ("fp32", "--fp32")):
        tag = f"check_{mode}"
        STATUS[tag] = "running" if images else "FAILED (no fixed_images.npy)"
        save_status()
        if not images:
            continue
        t = time.time()
        rc = sh(
            f"python scripts/platform_check.py --ckpt runs/teacher_b/ckpt/ema.safetensors --images {images[0]} "
            f"--out {OUT / 'check' / mode} {flags}",
            OUT / "logs" / f"{tag}.log",
            check=False,
        )
        STATUS[tag] = f"{'ok' if rc == 0 else f'FAILED ({rc})'} ({(time.time() - t) / 60:.1f} min)"
        ok &= rc == 0
        save_status()
    return ok


PART = 40 * 2**20  # the Colab transfer API sends a file as one base64 request; ~50 MB+ requests are refused


def split(src: Path, dl: Path, manifest: dict, name: str | None = None) -> None:
    """src -> dl/<name>.partNN (PART bytes each), recorded with its sha256 for reassembly."""
    h, parts, name = hashlib.sha256(), [], name or src.name
    with open(src, "rb") as f:
        while chunk := f.read(PART):
            h.update(chunk)
            parts.append(f"{name}.part{len(parts):02d}")
            (dl / parts[-1]).write_bytes(chunk)
    manifest[name] = {"sha256": h.hexdigest(), "bytes": src.stat().st_size, "parts": parts}


def pack() -> None:
    """Colab: results + logs + platform-check arrays as one archive, and the (real-run) GAN student weights, both
    split into PART-sized pieces under /content/dl with a MANIFEST.json (sha256) -- see scratchpad watchdog pull."""
    if KAGGLE:
        return
    dl = OUT.parent / "dl"
    shutil.rmtree(dl, ignore_errors=True)
    dl.mkdir()
    tgz, manifest = OUT.parent / "humandit_results.tar.gz", {}
    skip = (
        ".safetensors",
        "ffhq256_dinov2_real.npz",
    )  # weights travel separately; the npz is one of our inputs
    with tarfile.open(tgz, "w:gz") as t:
        t.add(OUT, arcname=OUT.name, filter=lambda ti: None if ti.name.endswith(skip) else ti)
    split(tgz, dl, manifest)
    tgz.unlink()
    for student in sorted((OUT / "results" / "runs").glob("student_*/student.safetensors")):
        if not student.parent.name.endswith("_toy"):
            split(student, dl, manifest, name=f"{student.parent.name}.safetensors")
    (dl / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    sizes = ", ".join(f"{k} ({v['bytes'] / 1e6:.1f} MB)" for k, v in manifest.items())
    print(f"[driver] packed {sizes}", flush=True)


def main() -> None:
    if STAGES == ["pack"]:  # re-pack an earlier run's outputs only
        pack()
        return
    setup()
    print(f"[driver] {'GPU' if GPU else 'TPU'}; stages {STAGES}", flush=True)
    try:
        if "toy" in STAGES:
            toy = run_stage(toy=True, which=MAIN)
            print(json.dumps(toy, indent=2), flush=True)
            if any(v != "ok" for v in toy.values()) or "--toy-only" in sys.argv:
                print("[driver] rehearsal did not fully pass (or --toy-only); stopping here", flush=True)
                return
        if "check" in STAGES:
            run_check()
        if MAIN in STAGES:
            shutil.rmtree(OUT / "results", ignore_errors=True)  # keep only real results
            run_stage(toy=False, which=MAIN)
        print(json.dumps(STATUS, indent=2), flush=True)
    finally:
        pack()
        STATUS["driver"] = "done"
        save_status()


if __name__ == "__main__":
    main()
