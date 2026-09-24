"""Operate the training pod from the laptop: launch, ship code, start the pipeline, watch, pull results.

    uv run python scripts/pod_ops.py launch --budget 5.5            # RTX 4090, Community, SSH via proxy
    uv run python scripts/pod_ops.py logs                           # tail the pipeline log
    uv run python scripts/pod_ops.py pull                           # download results over RunPod's HTTPS proxy
    uv run python scripts/pod_ops.py mark-pulled                    # lets the pod stop itself when finished

Two connection modes:
  * proxy (default): no public IP needed, so far more hosts are eligible. Uses RunPod's SSH gateway
    (<podHostId>@ssh.runpod.io, account-registered key). The gateway only offers interactive shells,
    so commands are piped through a pseudo-terminal and code ships as base64 text. Results come back
    through the HTTPS port proxy (https://<podId>-8000.proxy.runpod.net) from a short-lived server.
  * direct: public IP + mapped TCP port 22 (scp-free as well; same code paths).
State lives in .pod.json (git-ignored). The API key is never printed.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import secrets
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_ctl as rp

STATE = Path(".pod.json")
KEY = Path.home() / ".ssh" / "runpod_humandit_ed25519"
REMOTE = "/workspace/humandit"
SSH_OPTS = ["-i", str(KEY), "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30",
            "-o", "LogLevel=ERROR"]


def _state() -> dict:
    if not STATE.exists():
        sys.exit("no .pod.json; launch a pod first")
    return json.loads(STATE.read_text())


ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07")


def run(st: dict, script: str, timeout: int = 600) -> str:
    """Run a bash script on the pod; return only its output (gateway banner, prompts, echo stripped)."""
    tag = secrets.token_hex(4)
    begin, end = f"__BEGIN_{tag}__", f"__END_{tag}__"
    # printf splits each marker so the echoed command line never contains the full marker text
    body = f"printf '%s%s\\n' __BEGIN_ {tag}__\n{script}\nprintf '%s%s\\n' __END_ {tag}__"
    if st["mode"] == "proxy":
        cmd = ["ssh", "-tt", *SSH_OPTS, f"{st['host']}@ssh.runpod.io"]
        stdin = f"stty -echo 2>/dev/null\n{body}\nexit\n"
    else:
        cmd = ["ssh", *SSH_OPTS, "-p", str(st["port"]), f"root@{st['ip']}", "bash -s"]
        stdin = body
    proc = subprocess.run(cmd, input=stdin.encode(), capture_output=True, timeout=timeout, check=False)
    out = ANSI.sub("", (proc.stdout + proc.stderr).decode(errors="replace").replace("\r\n", "\n"))
    if begin in out:
        out = out[out.index(begin) + len(begin):]
        out = out[: out.index(end)] if end in out else out
    return out.strip("\n")


def _pod_host_id(pod_id: str) -> str | None:
    q = ("query P($input: PodFilter!) { pod(input: $input) { desiredStatus runtime { uptimeInSeconds } "
         "machine { podHostId } } }")
    pod = rp.gql(q, {"input": {"podId": pod_id}}).get("pod") or {}
    if not (pod.get("runtime") or {}).get("uptimeInSeconds"):
        return None
    return (pod.get("machine") or {}).get("podHostId")


def wait_ready(st: dict, timeout: int = 1200) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if st["mode"] == "proxy" and not st.get("host"):
            st["host"] = _pod_host_id(st["id"])
        elif st["mode"] == "direct" and not st.get("ip"):
            pod = rp._request("GET", f"{rp.REST}/pods/{st['id']}")
            ports = pod.get("portMappings") or {}
            if pod.get("publicIp") and "22" in ports:
                st["ip"], st["port"] = pod["publicIp"], ports["22"]
        if st.get("host") or st.get("ip"):
            marker = f"READY_{secrets.token_hex(4)}"
            try:
                if marker in run(st, f"echo {marker}", timeout=60):
                    return st
            except subprocess.TimeoutExpired:
                pass
        print(f"[pod] waiting for the pod to accept SSH... {int(time.time() - t0)}s", flush=True)
        time.sleep(20)
    sys.exit("pod did not become reachable over SSH")


def create(a: argparse.Namespace) -> dict:
    pub = Path(str(KEY) + ".pub").read_text().strip()
    proxy = a.mode == "proxy"
    body = {
        "name": "humandit", "imageName": rp.IMAGE, "gpuTypeIds": [a.gpu], "gpuCount": 1, "cloudType": a.cloud,
        "containerDiskInGb": a.container_disk, "volumeInGb": a.volume, "volumeMountPath": "/workspace",
        "ports": ["8000/http"] if proxy else ["22/tcp", "8000/http"], "supportPublicIp": not proxy,
        "minVCPUPerGPU": a.min_vcpu, "minRAMPerGPU": a.min_ram, "env": {"PUBLIC_KEY": pub}, "interruptible": False,
    }
    t0 = time.time()
    while True:  # Community stock fluctuates minute to minute; nothing is billed until creation succeeds
        try:
            pod = rp._request("POST", f"{rp.REST}/pods", body)
            break
        except SystemExit as e:
            if "no instances currently available" not in str(e) or time.time() - t0 > a.retry_min * 60:
                raise
            print(f"[pod] no {a.gpu} available yet; retrying ({int(time.time() - t0)}s)", flush=True)
            time.sleep(60)
    print(f"[pod] created {pod['id']} at ${pod.get('costPerHr')}/h ({a.mode} mode)", flush=True)
    st = {"id": pod["id"], "mode": a.mode, "usd_per_hour": float(pod.get("costPerHr") or a.usd_per_hour)}
    STATE.write_text(json.dumps(st, indent=2))
    return st


def check_host(st: dict, min_power_w: float) -> tuple[bool, str]:
    """Reject power-capped or already-busy GPUs (seen on Community hosts) before spending anything on them."""
    q = "nvidia-smi --query-gpu=name,power.limit,utilization.gpu,memory.used --format=csv,noheader,nounits"
    rows = [run(st, q), run(st, f"sleep 5; {q}")]
    try:
        vals = [[x.strip() for x in r.strip().splitlines()[-1].split(",")] for r in rows]
        name, power = vals[0][0], float(vals[0][1])
        util, mem = max(float(v[2]) for v in vals), max(float(v[3]) for v in vals)
    except (ValueError, IndexError):
        return False, f"could not parse nvidia-smi output: {rows}"
    msg = f"{name}: power limit {power:.0f} W, utilisation {util:.0f}%, memory in use {mem:.0f} MiB"
    healthy = power >= min_power_w and util <= 10 and mem <= 1024
    return healthy, msg


def launch(a: argparse.Namespace) -> None:
    if STATE.exists() and not a.force:
        sys.exit(".pod.json exists (a pod may already be running); use --force to launch another")
    for attempt in range(1, a.max_hosts + 1):
        st = wait_ready(create(a))
        STATE.write_text(json.dumps(st, indent=2))
        healthy, msg = check_host(st, a.min_power_w)
        print(f"[pod] host check ({attempt}/{a.max_hosts}): {msg} -> {'OK' if healthy else 'REJECTED'}", flush=True)
        if healthy:
            break
        if not a.auto_reject:
            rp._request("POST", f"{rp.REST}/pods/{st['id']}/stop")
            sys.exit("unhealthy host: pod stopped (not deleted); re-run with --auto-reject to replace it")
        rp._request("DELETE", f"{rp.REST}/pods/{st['id']}")  # brand-new pod: nothing on it yet
        STATE.unlink()
        print(f"[pod] terminated unhealthy pod {st['id']}; requesting another host", flush=True)
    else:
        sys.exit(f"no healthy host after {a.max_hosts} attempts")
    push(a)
    start(a)


def _code_tarball() -> bytes:
    """git HEAD without large binary figures (the pod only needs code and configs)."""
    raw = subprocess.run(["git", "archive", "--format=tar", "HEAD"], capture_output=True, check=True).stdout
    buf = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(raw)) as src, tarfile.open(fileobj=buf, mode="w:gz") as dst:
        for m in src.getmembers():
            if m.name.startswith(("docs/assets/", "legacy/images/")):
                continue
            dst.addfile(m, src.extractfile(m) if m.isfile() else None)
    return buf.getvalue()


def push(_: argparse.Namespace) -> None:
    st = _state()
    b64 = base64.encodebytes(_code_tarball()).decode()  # 76-char lines: safe for a pseudo-terminal
    ok = f"SHIP_OK_{secrets.token_hex(4)}"
    script = (f"mkdir -p {REMOTE} && base64 -d > /tmp/code.tgz <<'B64EOF'\n{b64}B64EOF\n"
              f"tar xzf /tmp/code.tgz -C {REMOTE} && echo {ok}")
    out = run(st, script, timeout=900)
    if ok not in out:
        sys.exit(f"code upload failed:\n{out[-2000:]}")
    print(f"[pod] shipped {len(b64) / 1e3:.0f} kB (base64) of code (git HEAD) to {REMOTE}", flush=True)


def start(a: argparse.Namespace) -> None:
    st = _state()
    env = (f"BUDGET_USD={a.budget} USD_PER_HOUR={st['usd_per_hour'] + a.disk_usd_per_hour:.3f} "
           f"STUDENT_MIN={a.student_min} AUTO_STOP={int(a.auto_stop)} GRACE_MIN={a.grace_min}")
    ok = f"STARTED_{secrets.token_hex(4)}"
    out = run(st, f"cd {REMOTE} && setsid nohup env {env} bash scripts/pod/run_all.sh "
                  f"> /workspace/run_all.log 2>&1 < /dev/null & sleep 3; pgrep -f run_all.sh >/dev/null && echo {ok}")
    if ok not in out:
        sys.exit(f"pipeline did not start:\n{out[-2000:]}")
    print(f"[pod] pipeline started ({env}); follow with: uv run python scripts/pod_ops.py logs", flush=True)


def logs(a: argparse.Namespace) -> None:
    out = run(_state(), f"tail -n {a.n} /workspace/run_all.log; echo; nvidia-smi --query-gpu=utilization.gpu,"
                        "memory.used,temperature.gpu --format=csv,noheader; echo stages done: $(ls /workspace/state)")
    print(out.strip())


def pull(a: argparse.Namespace) -> None:
    st = _state()
    token = secrets.token_hex(16)  # unguessable file name while the temporary server is up
    ok = f"SERVING_{secrets.token_hex(4)}"
    run(st, (f"mkdir -p /workspace/share && cd /workspace && tar czf /workspace/share/{token}.tgz "
             "--ignore-failed-read runs/eval/*.json runs/eval/*.png runs/rae_gate/metrics.json "
             "runs/rae_gate/recon_grid.png runs/attributes runs/*/ckpt/ema.safetensors runs/*/student.safetensors "
             "runs/*/samples/*.png runs/*/metrics.jsonl runs/*/train_summary.json runs/*/summary.json logs "
             f"2>/dev/null; cd /workspace/share && (setsid nohup python3 -m http.server 8000 >/dev/null 2>&1 &) ; "
             f"sleep 2; echo {ok}"), timeout=1800)
    url = f"https://{st['id']}-8000.proxy.runpod.net/{token}.tgz"
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / "pod_results.tgz"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "humandit-pod-ops/0.1"})
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 22):
                f.write(chunk)
    finally:
        run(st, "pkill -f 'http.server 8000'; rm -rf /workspace/share")
    with tarfile.open(tmp) as t:
        t.extractall(dest, filter="data")
    print(f"[pod] pulled {tmp.stat().st_size / 1e6:.1f} MB into {dest}/ (temporary server stopped)", flush=True)
    tmp.unlink()


def mark_pulled(_: argparse.Namespace) -> None:
    run(_state(), "touch /workspace/state/pulled")
    print("[pod] marked as pulled; the pod will stop itself after its pipeline finishes")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    la = sub.add_parser("launch")
    la.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
    la.add_argument("--cloud", default="COMMUNITY", choices=["COMMUNITY", "SECURE"])
    la.add_argument("--mode", default="proxy", choices=["proxy", "direct"])
    la.add_argument("--budget", type=float, required=True, help="total USD for the whole pipeline")
    la.add_argument("--usd-per-hour", type=float, default=0.34)
    la.add_argument("--disk-usd-per-hour", type=float, default=0.03, help="storage cost added to the GPU rate")
    la.add_argument("--student-min", type=float, default=45)
    la.add_argument("--container-disk", type=int, default=130)
    la.add_argument("--volume", type=int, default=40)
    la.add_argument("--min-vcpu", type=int, default=6)
    la.add_argument("--min-ram", type=int, default=24)
    la.add_argument("--retry-min", type=float, default=30)
    la.add_argument("--auto-stop", action="store_true", default=True)
    la.add_argument("--grace-min", type=int, default=60)
    la.add_argument("--force", action="store_true")
    la.add_argument("--min-power-w", type=float, default=300, help="reject GPUs power-capped below this")
    la.add_argument("--auto-reject", action="store_true", help="terminate unhealthy fresh pods and retry")
    la.add_argument("--max-hosts", type=int, default=4)
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
