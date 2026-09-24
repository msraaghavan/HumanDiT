"""Minimal RunPod controller (REST v1 + GraphQL) for the training pod.

The API key is read from $RUNPOD_API_KEY or ~/.runpod/api_key and is never printed or logged.

    uv run python scripts/runpod_ctl.py balance
    uv run python scripts/runpod_ctl.py gpus
    uv run python scripts/runpod_ctl.py create --gpu "NVIDIA GeForce RTX 4090" --cloud COMMUNITY
    uv run python scripts/runpod_ctl.py status <pod_id>
    uv run python scripts/runpod_ctl.py stop <pod_id>
    uv run python scripts/runpod_ctl.py terminate <pod_id> --yes
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REST = "https://rest.runpod.io/v1"
GRAPHQL = "https://api.runpod.io/graphql"
IMAGE = "runpod/pytorch:1.3.2-cu1281-torch2130-ubuntu2404"
CANDIDATE_GPUS = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 5090",
    "NVIDIA RTX A6000",
    "NVIDIA L40S",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA H100 80GB HBM3",
]


def _key() -> str:
    """$RUNPOD_API_KEY if it looks like a real key (rpa_...), else ~/.runpod/api_key. Never printed."""
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if key.startswith("rpa_"):
        return key
    path = Path.home() / ".runpod" / "api_key"
    if not path.exists():
        sys.exit("No RunPod API key: create ~/.runpod/api_key yourself (or set RUNPOD_API_KEY).")
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16"):
        try:
            return raw.decode(enc).strip()
        except UnicodeDecodeError:
            continue
    sys.exit("~/.runpod/api_key is not valid text")


def _request(method: str, url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {_key()}")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "humandit-runpod-ctl/0.1 (+python-urllib)")  # Cloudflare blocks the default UA
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            txt = r.read().decode()
            return json.loads(txt) if txt else {}
    except urllib.error.HTTPError as e:
        sys.exit(f"RunPod API error {e.code}: {e.read().decode()[:500]}")


def gql(query: str, variables: dict | None = None) -> dict:
    out = _request("POST", GRAPHQL, {"query": query, "variables": variables or {}})
    if "errors" in out:
        sys.exit(f"GraphQL error: {out['errors']}")
    return out["data"]


def cmd_balance(_: argparse.Namespace) -> None:
    me = gql("query { myself { clientBalance currentSpendPerHr spendLimit } }")["myself"]
    print(json.dumps(me, indent=2))


def cmd_gpus(_: argparse.Namespace) -> None:
    rows = gql(
        "query { gpuTypes { id displayName memoryInGb communityPrice securePrice "
        "lowestPrice(input: {gpuCount: 1}) { stockStatus } } }"
    )["gpuTypes"]
    for r in rows:
        if r["id"] in CANDIDATE_GPUS:
            stock = (r.get("lowestPrice") or {}).get("stockStatus")
            print(f"{r['id']:<28} {r['memoryInGb']:>3} GB  community ${r['communityPrice'] or 0:.2f}/h  "
                  f"secure ${r['securePrice'] or 0:.2f}/h  stock={stock}")


def cmd_add_ssh_key(a: argparse.Namespace) -> None:
    """Append a public key to the account's SSH keys (same logic as runpodctl's AddPublicSSHKey)."""
    new = Path(a.pubkey).expanduser().read_text().strip()
    raw = gql("query { myself { pubKey } }")["myself"].get("pubKey") or ""
    if any(line.strip() == new for line in raw.splitlines()):
        print("key already registered")
        return
    sep = "\n\n"  # runpodctl joins account keys with a blank line
    keys = (raw.strip() + sep if raw.strip() else "") + new
    gql("mutation M($input: UpdateUserSettingsInput) { updateUserSettings(input: $input) { id } }",
        {"input": {"pubKey": keys}})
    n = len([line for line in keys.splitlines() if line.strip()])
    print(f"added {new.split()[-1]} ({n} account keys total; existing keys kept)")


def cmd_create(a: argparse.Namespace) -> None:
    pub = Path(a.ssh_pubkey).expanduser().read_text().strip()
    body = {
        "name": a.name,
        "imageName": IMAGE,
        "gpuTypeIds": [a.gpu],
        "gpuCount": 1,
        "cloudType": a.cloud,
        "containerDiskInGb": a.container_disk,
        "volumeInGb": a.volume,
        "volumeMountPath": "/workspace",
        "ports": ["22/tcp"],
        "supportPublicIp": True,
        "minVCPUPerGPU": 8,
        "minRAMPerGPU": 32,
        "env": {"PUBLIC_KEY": pub},  # RunPod images install this into authorized_keys
        "interruptible": False,
    }
    pod = _request("POST", f"{REST}/pods", body)
    print(json.dumps({k: pod.get(k) for k in ("id", "costPerHr", "desiredStatus", "machine")}, indent=2))


def cmd_status(a: argparse.Namespace) -> None:
    for _ in range(a.wait // 10 + 1):
        pod = _request("GET", f"{REST}/pods/{a.pod_id}")
        ip, ports = pod.get("publicIp"), pod.get("portMappings") or {}
        if (ip and "22" in ports) or not a.wait:
            break
        time.sleep(10)
    info = {k: pod.get(k) for k in ("id", "desiredStatus", "costPerHr", "publicIp", "portMappings", "lastStartedAt")}
    print(json.dumps(info, indent=2))
    if pod.get("publicIp") and "22" in (pod.get("portMappings") or {}):
        print(f"ssh -p {pod['portMappings']['22']} root@{pod['publicIp']}")


def cmd_stop(a: argparse.Namespace) -> None:
    print(json.dumps(_request("POST", f"{REST}/pods/{a.pod_id}/stop"), indent=2))


def cmd_terminate(a: argparse.Namespace) -> None:
    if not a.yes:
        sys.exit("Terminating deletes the pod and its disks. Re-run with --yes to confirm.")
    _request("DELETE", f"{REST}/pods/{a.pod_id}")
    print(f"terminated {a.pod_id}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("balance").set_defaults(fn=cmd_balance)
    sub.add_parser("gpus").set_defaults(fn=cmd_gpus)
    k = sub.add_parser("add-ssh-key")
    k.add_argument("--pubkey", default="~/.ssh/runpod_humandit_ed25519.pub")
    k.set_defaults(fn=cmd_add_ssh_key)
    c = sub.add_parser("create")
    c.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
    c.add_argument("--cloud", default="COMMUNITY", choices=["COMMUNITY", "SECURE"])
    c.add_argument("--name", default="humandit")
    c.add_argument("--container-disk", type=int, default=160)
    c.add_argument("--volume", type=int, default=40)
    c.add_argument("--ssh-pubkey", default="~/.ssh/runpod_humandit_ed25519.pub")
    c.set_defaults(fn=cmd_create)
    for name, fn in (("status", cmd_status), ("stop", cmd_stop), ("terminate", cmd_terminate)):
        p = sub.add_parser(name)
        p.add_argument("pod_id")
        if name == "status":
            p.add_argument("--wait", type=int, default=0, help="seconds to wait for SSH to come up")
        if name == "terminate":
            p.add_argument("--yes", action="store_true")
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
