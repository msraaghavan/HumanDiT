"""Start the driver on a Colab VM as a detached process and return at once.

    colab upload -s S scripts/colab/driver.py /content/driver.py
    colab exec -s S --env HUMANDIT_STAGES=fix -f scripts/colab/launch.py

`colab exec -f driver.py` runs the driver inside the kernel, tied to the CLI's connection: when the CLI's login lapsed
mid-run, the stream died and Colab reclaimed the VM ~20 min later. Detached, the driver outlives the connection; its
output goes to /content/driver.log and its progress to /content/humandit_out/status.json, and the watchdog keeps the
idle kernel alive while the driver runs.
"""

import os
import subprocess

log = open("/content/driver.log", "a")  # noqa: SIM115
proc = subprocess.Popen(
    ["python3", "-u", "/content/driver.py"],
    env={**os.environ, "HUMANDIT_STAGES": os.environ.get("HUMANDIT_STAGES", "toy,real")},
    stdout=log,
    stderr=subprocess.STDOUT,
    start_new_session=True,  # survives the kernel and the CLI connection
    cwd="/content",
)
print(f"driver started, pid {proc.pid}, stages {os.environ.get('HUMANDIT_STAGES')}, log /content/driver.log")
