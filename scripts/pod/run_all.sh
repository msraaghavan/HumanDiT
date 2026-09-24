#!/usr/bin/env bash
# Idempotent end-to-end pipeline for the RunPod GPU. Each stage writes a marker and is skipped
# on re-run, so the script can be restarted after any interruption.
#   BUDGET_USD=3.0 USD_PER_HOUR=0.34 bash scripts/pod/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FAST=/root/hd            # container disk: datasets, latent cache, weights (large, re-creatable)
KEEP=/workspace          # volume: runs, logs, markers (small, persistent)
mkdir -p "$FAST"/{data,cache,pretrained} "$KEEP"/{runs,logs,state}
for d in data cache pretrained; do [ -L "$d" ] || { rm -rf "$d"; ln -s "$FAST/$d" "$d"; }; done
[ -L runs ] || { rm -rf runs; ln -s "$KEEP/runs" runs; }
BUDGET_USD=${BUDGET_USD:-3.0}; USD_PER_HOUR=${USD_PER_HOUR:-0.34}; NW=$(( $(nproc) > 12 ? 12 : $(nproc) ))

stage() {  # stage <name> <command...>
  local name=$1; shift
  if [ -f "$KEEP/state/$name.done" ]; then echo "[run_all] skip $name"; return; fi
  echo "[run_all] >>> $name  ($(date -u +%H:%M:%S))"
  "$@" 2>&1 | tee -a "$KEEP/logs/$name.log"
  touch "$KEEP/state/$name.done"
}

stage setup bash scripts/pod/setup.sh
stage ffhq uv run python -u scripts/prepare_data.py ffhq --workers "$NW" --batch 64
stage celeba uv run python -u scripts/prepare_data.py celeba --workers "$NW" --batch 128
stage attributes bash -c "uv run python -u scripts/build_attributes.py && mkdir -p runs/attributes && cp cache/attribute_probe.pt cache/ffhq256/attributes.json runs/attributes/"
stage rae_gate uv run python -u scripts/rae_reconstruction_gate.py --n 10000 --batch 64 --rfid --out runs/rae_gate
stage teacher uv run python -u scripts/train.py configs/pod_teacher_b.yaml \
  budget.budget_usd="$BUDGET_USD" budget.usd_per_hour="$USD_PER_HOUR" data.num_workers="$NW"
echo "[run_all] teacher pipeline complete ($(date -u +%H:%M:%S))"
