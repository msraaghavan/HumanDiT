"""Operate the training pod from the laptop: launch, ship code, start the pipeline, watch, pull results.

    uv run python scripts/pod_ops.py launch --gpu "NVIDIA GeForce RTX 4090" --cloud COMMUNITY --budget 4.0
    uv run python scripts/pod_ops.py logs            # tail the pipeline log
    uv run python scripts/pod_ops.py pull            # copy small results (evals, EMA weights, samples, logs)
    uv run python scripts/pod_ops.py mark-pulled     # lets the pod stop itself after the final pull
State (pod id, ssh endpoint) lives in .pod.json (git-ignored). The API key is never printed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_ctl as rp

STATE = Path(".pod.json")
KEY = Path.home() / ".ssh" / "runpod_humandit_ed25519"
REMOTE = "/workspace/humandit"


def _state() -> dict:
    if not STATE.exists():
        sys.exit("no .pod.json; launch a pod first")
    return json.loads(STATE.read_text())


def _ssh_base(st: dict) -> list[str]:
    return ["ssh", "-i", str(KEY), "-p", str(st["port"]), "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ServerAliveInterval=30", f"root@{st['ip']}"]


def ssh(st: dict, cmd: str, stdin: bytes | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run([*_ssh_base(st), cmd], input=stdin, check=check)


def wait_ssh(pod_id: str, timeout: int = 900) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        pod = rp._request("GET", f"{rp.REST}/pods/{pod_id}")
        ip, ports = pod.get("publicIp"), pod.get("portMappings") or {}
        if ip and "22" in ports:
            st = {"id": pod_id, "ip": ip, "port": ports["22"], "cost_per_hr": pod.get("costPerHr")}
            if subprocess.run([*_ssh_base(st), "echo ok"], capture_output=True, timeout=30).returncode == 0:
                return st
        print(f"[pod] waiting for SSH... {int(time.time() - t0)}s", flush=True)
        time.sleep(15)
    sys.exit("pod did not become reachable over SSH")


def launch(a: argparse.Namespace) -> None:
    if STATE.exists() and not a.force:
        sys.exit(".pod.json exists (a pod may already be running); use --force to launch another")
    pub = Path(str(KEY) + ".pub").read_text().strip()  # RunPod images add $PUBLIC_KEY to authorized_keys
    pod = rp._request("POST", f"{rp.REST}/pods", {
        "name": "humandit", "imageName": rp.IMAGE, "gpuTypeIds": [a.gpu], "gpuCount": 1, "cloudType": a.cloud,
        "containerDiskInGb": a.container_disk, "volumeInGb": a.volume, "volumeMountPath": "/workspace",
        "ports": ["22/tcp"], "supportPublicIp": True, "minVCPUPerGPU": 8, "minRAMPerGPU": 32,
        "env": {"PUBLIC_KEY": pub}, "interruptible": False})
    print(f"[pod] created {pod['id']} at ${pod.get('costPerHr')}/h", flush=True)
    st = wait_ssh(pod["id"])
    st["usd_per_hour"] = float(pod.get("costPerHr") or a.usd_per_hour)
    STATE.write_text(json.dumps(st, indent=2))
    push(a)
    start(a)


def push(_: argparse.Namespace) -> None:
    st = _state()
    tar = subprocess.run(["git", "archive", "--format=tar", "HEAD"], capture_output=True, check=True).stdout
    ssh(st, f"mkdir -p {REMOTE} && tar -x -C {REMOTE}", stdin=tar)
    print(f"[pod] shipped {len(tar) / 1e6:.1f} MB of code (git HEAD) to {REMOTE}", flush=True)


def start(a: argparse.Namespace) -> None:
    st = _state()
    env = (f"BUDGET_USD={a.budget} USD_PER_HOUR={st['usd_per_hour']} STUDENT_MIN={a.student_min} "
           f"AUTO_STOP={int(a.auto_stop)} GRACE_MIN={a.grace_min}")
    ssh(st, f"cd {REMOTE} && nohup env {env} bash scripts/pod/run_all.sh > /workspace/run_all.log 2>&1 &")
    print(f"[pod] pipeline started ({env}); follow with: uv run python scripts/pod_ops.py logs", flush=True)


def logs(a: argparse.Namespace) -> None:
    ssh(_state(), f"tail -n {a.n} /workspace/run_all.log; nvidia-smi --query-gpu=utilization.gpu,memory.used "
                  "--format=csv,noheader; ls /workspace/state", check=False)


def pull(a: argparse.Namespace) -> None:
    st = _state()
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    remote = ("cd /workspace && tar czf - --ignore-failed-read runs/eval/*.json runs/eval/*.png runs/eval/*.npz "
              "runs/rae_gate/metrics.json runs/rae_gate/recon_grid.png runs/attributes "
              "runs/*/ckpt/ema.safetensors runs/*/student.safetensors runs/*/samples runs/*/metrics.jsonl "
              "runs/*/train_summary.json runs/*/summary.json logs 2>/dev/null")
    proc = subprocess.run([*_ssh_base(st), remote], capture_output=True, check=False)
    subprocess.run(["tar", "xzf", "-", "-C", str(dest)], input=proc.stdout, check=True)
    print(f"[pod] pulled {len(proc.stdout) / 1e6:.1f} MB into {dest}/", flush=True)


def mark_pulled(_: argparse.Namespace) -> None:
    ssh(_state(), "touch /workspace/state/pulled")
    print("[pod] marked as pulled; the pod will stop itself after its pipeline finishes")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    la = sub.add_parser("launch")
    la.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
    la.add_argument("--cloud", default="COMMUNITY", choices=["COMMUNITY", "SECURE"])
    la.add_argument("--budget", type=float, required=True, help="total USD for the whole pipeline")
    la.add_argument("--usd-per-hour", type=float, default=0.34)
    la.add_argument("--student-min", type=float, default=45)
    la.add_argument("--container-disk", type=int, default=160)
    la.add_argument("--volume", type=int, default=40)
    la.add_argument("--auto-stop", action="store_true", default=True)
    la.add_argument("--grace-min", type=int, default=60)
    la.add_argument("--force", action="store_true")
    la.set_defaults(fn=launch)
    sub.add_parser("push").set_defaults(fn=push)
    lg = sub.add_parser("logs")
    lg.add_argument("-n", type=int, default=40)
    lg.set_defaults(fn=logs)
    pl = sub.add_parser("pull")
    pl.add_argument("--dest", default=".")
    pl.set_defaults(fn=pull)
    sub.add_parser("mark-pulled").set_defaults(fn=mark_pulled)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
