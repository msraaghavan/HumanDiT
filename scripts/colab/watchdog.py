#!/usr/bin/env python3
"""Colab watchdog for the HumanDiT runs (run it on the machine that drives the Colab CLI; WSL on Windows). Every 5 minutes: balance + spend, every session's status (IDLE/BUSY, age)
and its driver's job status. Guards (results are always pulled before a VM is stopped):
  * idle   a session IDLE for IDLE_MIN minutes is stopped -- unless held (touch ~/.humandit_watch/hold/<name> while
           uploading / setting up; a hold expires HOLD_MIN minutes after its last touch)
  * budget once BUDGET_CU compute units have been spent since the watchdog started, every session is stopped and
           the watchdog exits (more spend needs the user's OK)
  * age    any session older than MAX_AGE_H hours is stopped
Log: runs/colab/watchdog.log (and stdout); pulled results: runs/colab/pulled/<session>/. Exits when no sessions have been left for two checks.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

INTERVAL = 300
IDLE_MIN = 15
HOLD_MIN = 40
BUDGET_CU = float(os.environ.get("BUDGET_CU", "60"))
MAX_AGE_H = float(os.environ.get("MAX_AGE_H", "8"))
COLAB = shutil.which("colab") or str(Path.home() / ".local/bin/colab")
HOME = Path.home() / ".humandit_watch"
HOLD = HOME / "hold"
STATE = HOME / "state.json"
BASE = Path(__file__).resolve().parents[2] / "runs" / "colab"
PULLED = BASE / "pulled"
LOG = BASE / "watchdog.log"
for d in (HOLD, PULLED, BASE):
    d.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def colab(*args: str, timeout: int = 180) -> tuple[int, str]:
    try:
        r = subprocess.run([COLAB, *args], capture_output=True, text=True, timeout=timeout)
        out = r.stdout + r.stderr
        return r.returncode, re.sub(r"token=[^ &\s]+", "token=REDACTED", out)
    except subprocess.TimeoutExpired:
        return 124, f"timeout: colab {' '.join(args)}"


def usage() -> tuple[float | None, float | None]:
    _, out = colab("usage")
    b = re.search(r"balance:\s*([\d.]+)", out)
    r = re.search(r"Usage rate:\s*([\d.]+)", out)
    return (float(b.group(1)) if b else None), (float(r.group(1)) if r else None)


def sessions() -> list[str]:
    _, out = colab("sessions")
    # session lines look like "[name] <assignment> | Hardware: ..."; the CLI's own "[colab] ..." messages do not
    return [m for m in re.findall(r"^\[([^\]]+)\].*\| Hardware:", out, re.M) if m != "?"]


def session_status(name: str) -> tuple[str, str]:
    _, out = colab("status", "-s", name)
    st = re.search(r"Status:\s*(\w+)", out)
    hw = re.search(r"Hardware:\s*(\w+)", out)
    return (st.group(1) if st else "?"), (hw.group(1) if hw else "?")


def ls(name: str, path: str) -> list[str]:
    rc, out = colab("ls", "-s", name, path)
    return [] if rc or "not found" in out.lower() else [x.strip() for x in out.splitlines() if x.strip()]


def download(name: str, remote: str, local: Path) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    rc, out = colab("download", "-s", name, remote, str(local), timeout=900)
    return rc == 0 and local.exists() and "fail" not in out.lower()


def driver_status(name: str) -> dict:
    tmp = HOME / f"status_{name}.json"
    if "status.json" in ls(name, "/content/humandit_out") and download(
        name, "/content/humandit_out/status.json", tmp
    ):
        try:
            return json.loads(tmp.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def summarize(st: dict) -> str:
    if not st:
        return "driver: not started"
    running = [k for k, v in st.items() if v == "running"]
    failed = [k for k, v in st.items() if str(v).startswith("FAILED")]
    ok = sum(str(v).startswith("ok") for v in st.values())
    s = f"jobs ok {ok}" + (f", RUNNING {running[0]}" if running else "")
    s += f", FAILED {failed}" if failed else ""
    return s + (", driver DONE" if st.get("driver") == "done" else "")


def pull(name: str) -> bool:
    """Download /content/dl (the driver's packed parts + MANIFEST.json), then reassemble each file and verify its
    sha256. True if everything is local and verified (or there is nothing to pull)."""
    import hashlib

    files = ls(name, "/content/dl")
    if not files:
        return True
    dst = PULLED / name
    parts = dst / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    manifest, fresh = parts / "MANIFEST.json", parts / "MANIFEST.new"
    if not download(name, "/content/dl/MANIFEST.json", fresh):
        return False
    if not manifest.exists() or manifest.read_bytes() != fresh.read_bytes():
        for stale in parts.glob("*.part*"):  # parts of an earlier pack on this VM reuse the same names
            stale.unlink()
    fresh.replace(manifest)
    ok = True
    for f in files:
        if f != "MANIFEST.json" and not (parts / f).exists():
            ok &= download(name, f"/content/dl/{f}", parts / f)
    if not ok:
        return False

    def sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            while chunk := fh.read(1 << 24):
                h.update(chunk)
        return h.hexdigest()

    for fname, info in json.loads(manifest.read_text()).items():
        out = dst / fname
        if out.exists() and sha256(out) == info["sha256"]:
            continue
        h = hashlib.sha256()
        with open(out, "wb") as w:
            for p in info["parts"]:
                b = (parts / p).read_bytes()
                h.update(b)
                w.write(b)
        if h.hexdigest() != info["sha256"]:
            out.unlink()
            for part in info["parts"]:  # a bad or stale part: fetch them all again on the next pull
                (parts / part).unlink(missing_ok=True)
            log(f"  {name}: {fname} sha256 MISMATCH after reassembly")
            ok = False
        else:
            log(f"  {name}: pulled {fname} ({info['bytes'] / 1e6:.1f} MB, sha256 ok)")
    return ok


def stop(name: str, why: str, force: bool = False) -> None:
    pulled = pull(name)
    if not pulled and not force:
        log(f"  {name}: {why}, but pulling results FAILED -- not stopping; will retry")
        return
    colab("stop", "-s", name)
    log(f"  {name}: STOPPED ({why}); results {'pulled' if pulled else 'NOT fully pulled'} -> {PULLED / name}")


def main() -> None:
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    bal, _ = usage()
    state.setdefault("start_balance", bal)
    state.setdefault("first_seen", {})
    state.setdefault("idle_since", {})
    log(
        f"watchdog up: balance {bal} CU (start {state['start_balance']}), budget {BUDGET_CU} CU, "
        f"idle stop {IDLE_MIN} min, max age {MAX_AGE_H} h"
    )
    empty = 0
    while True:
        now = time.time()
        bal, rate = usage()
        spent = (state["start_balance"] - bal) if bal is not None and state["start_balance"] else 0.0
        names = sessions()
        log(
            f"balance {bal} CU | spent {spent:.2f} / {BUDGET_CU} | rate {rate}/h | sessions {names or 'none'}"
        )
        if spent >= BUDGET_CU:
            log(f"BUDGET REACHED ({spent:.2f} CU) -- stopping every session; more spend needs the user's OK")
            for n in names:
                stop(n, "budget", force=True)
            break
        for n in names:
            state["first_seen"].setdefault(n, now)
            age_h = (now - state["first_seen"][n]) / 3600
            st, hw = session_status(n)
            drv = driver_status(n)
            hold = HOLD / n
            held = hold.exists() and now - hold.stat().st_mtime < HOLD_MIN * 60
            if st == "IDLE" and not held:
                state["idle_since"].setdefault(n, now)
            else:
                state["idle_since"].pop(n, None)
            idle_min = (now - state["idle_since"][n]) / 60 if n in state["idle_since"] else 0
            log(
                f"  {n} [{hw}] {st}{' (held)' if held else ''} age {age_h:.2f} h"
                f"{f' idle {idle_min:.0f} min' if idle_min else ''} | {summarize(drv)}"
            )
            if age_h >= MAX_AGE_H:
                stop(n, f"max age {MAX_AGE_H} h", force=True)
            elif idle_min >= IDLE_MIN:
                stop(n, f"idle {idle_min:.0f} min")
        STATE.write_text(json.dumps(state))
        empty = empty + 1 if not names else 0
        if empty >= 2:
            log("no sessions left -- watchdog exiting")
            break
        time.sleep(INTERVAL)


if __name__ == "__main__":
    sys.exit(main())
