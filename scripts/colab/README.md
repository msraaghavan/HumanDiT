# Colab runs (how the follow-ups were produced)

The follow-up jobs ran on a Colab A100. They were: the fixed GAN student, its FID-50k, attribute control and a
step sweep at the tuned guidance scale, and latent statistics. The GPU was picked by a measured bake-off; see
[docs/compute_platforms.md](../../docs/compute_platforms.md). Everything here runs from a Linux, macOS or WSL shell,
because the Colab CLI has no Windows build.

```bash
uv tool install google-colab-cli                   # then any colab command opens the one-time Google login
uv run python scripts/colab/stage_inputs.py         # runs/colab/inputs: git-archive of HEAD + weights + fixed images

colab new -s run --gpu A100                         # 5.3 CU/h (measured)
python3 scripts/colab/watchdog.py &                 # every 5 min: spend, VM state, job status; idle/budget/age stops
bash scripts/colab/upload_inputs.sh run             # 40 MB parts, sha256-checked on the VM
colab exec -s run --timeout 36000 --env HUMANDIT_STAGES=toy,real -f scripts/colab/driver.py
python3 scripts/colab/pull.py run                   # runs/colab/pulled/run/ (the watchdog also pulls before stopping)
colab stop -s run
```

- **`driver.py`:** installs missing packages (pinned against the image's torch). It rehearses every job at toy size
  and stops if any fails, then runs the real jobs. Status goes to `/content/humandit_out/status.json`. Results are
  packed as 40 MB parts plus a sha256 manifest under `/content/dl`.
- **Bake-off:** `HUMANDIT_STAGES=toy,check` adds `scripts/platform_check.py`, an agreement and speed check against an
  fp32 reference.
- **TPU (`--tpu`, torch_xla):** unverified. Its first real run hit two known bugs, left unfixed on purpose; see the
  platform doc.
- **Watchdog guards:**
  - Idle: stops a VM idle for 15 min. A VM is protected while `~/.humandit_watch/hold/<session>` was touched within
    the last 40 min; `upload_inputs.sh` does this.
  - Budget: stops everything at `BUDGET_CU` (default 60).
  - Age: stops any VM at `MAX_AGE_H` (default 8).
  - It never stops a VM whose results failed to pull, except at the budget or age limit.
