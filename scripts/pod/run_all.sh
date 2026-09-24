#!/usr/bin/env bash
# Idempotent end-to-end pipeline for one RunPod GPU. Each stage writes a marker and is skipped on
# re-run, so the script can be restarted after any interruption.
#   BUDGET_USD=4.0 USD_PER_HOUR=0.34 STUDENT_MIN=45 AUTO_STOP=1 bash scripts/pod/run_all.sh
#   DRY_RUN=1 bash scripts/pod/run_all.sh     # tiny end-to-end run on any machine (laptop check)
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DRY=${DRY_RUN:-0}
if [ "$DRY" = 1 ]; then
  FAST=${FAST:-.dryrun}; KEEP=${KEEP:-.dryrun}; DATA=${DATA:-data}
  export HUMANGEN_PRETRAINED=${HUMANGEN_PRETRAINED:-pretrained/raev2}
else
  FAST=${FAST:-/root/hd}; KEEP=${KEEP:-/workspace}; DATA=${DATA:-$FAST/data}
  export HUMANGEN_PRETRAINED=${HUMANGEN_PRETRAINED:-$FAST/pretrained/raev2}
fi
CACHE=$FAST/cache; RUNS=$KEEP/runs; LOGS=$KEEP/logs; STATE=$KEEP/state
mkdir -p "$DATA" "$CACHE" "$RUNS" "$LOGS" "$STATE"
BUDGET_USD=${BUDGET_USD:-4.0}; USD_PER_HOUR=${USD_PER_HOUR:-0.34}; STUDENT_MIN=${STUDENT_MIN:-45}
NW=$(( $(nproc) > 12 ? 12 : $(nproc) ))
[ -f "$STATE/start_epoch" ] || date +%s > "$STATE/start_epoch"
PY="uv run python -u"

if [ "$DRY" = 1 ]; then  # tiny sizes: every code path runs, nothing is meaningful
  FFHQ_ARGS="--shards 1 --limit 96 --batch 8 --workers 0"; CELEBA_ARGS="--shards 1 --limit 160 --batch 8 --workers 0"
  ATTR_ARGS="--min-bacc 0.5 --max-k 4 --min-count 3"; GATE_ARGS="--n 8 --batch 4"
  TEACHER_ARGS="model.preset=T train.batch_size=8 train.max_steps=30 train.precision=fp32 data.num_workers=0 log.every=10 log.sample_every=1000 log.decoder_device=cpu log.sample_steps=4"
  EVAL_ARGS="--n 48 --batch 16 --steps 4 --control-n 16 --real-limit 64 --metrics fid,fd_dinov2,pr,control,memo"
  STUDENT_ARGS="--minutes 0.5 --groups 2 --gen-per-group 4 --pos-per-group 4 --gan-batch 4 --sample-every 1000"
  STUDENT_MIN=0.5
else
  FFHQ_ARGS="--workers $NW --batch 64"; CELEBA_ARGS="--workers $NW --batch 128"
  ATTR_ARGS=""; GATE_ARGS="--n 10000 --batch 64 --rfid"
  TEACHER_ARGS="data.num_workers=$NW"
  EVAL_ARGS="--n 50000 --batch 250 --steps 50"
  STUDENT_ARGS="--minutes $STUDENT_MIN"
fi

stage() {  # stage <name> <command...>
  local name=$1; shift
  if [ -f "$STATE/$name.done" ]; then echo "[run_all] skip $name"; return; fi
  echo "[run_all] >>> $name  ($(date -u +%H:%M:%S))"
  "$@" 2>&1 | tee -a "$LOGS/$name.log"
  touch "$STATE/$name.done"
}

teacher_hours() {  # remaining budget minus time reserved for the stages after the teacher
  uv run python - "$BUDGET_USD" "$USD_PER_HOUR" "$STATE/start_epoch" "$STUDENT_MIN" <<'PY'
import sys, time
budget, rate, start, smin = float(sys.argv[1]), float(sys.argv[2]), float(open(sys.argv[3]).read()), float(sys.argv[4])
used = (time.time() - start) / 3600
reserve = 1.0 + 2 * smin / 60 + 2 * 0.3 + 0.2  # teacher eval + 2 students + 2 student evals + packaging
print(f"{max(budget / rate - used - reserve, 0.05):.3f}")
PY
}

[ "$DRY" = 1 ] || stage setup bash scripts/pod/setup.sh
stage ffhq $PY scripts/prepare_data.py ffhq --data-dir "$DATA" --cache-dir "$CACHE" $FFHQ_ARGS
stage celeba $PY scripts/prepare_data.py celeba --data-dir "$DATA" --cache-dir "$CACHE" $CELEBA_ARGS
stage attributes bash -c "$PY scripts/build_attributes.py --cache '$CACHE' $ATTR_ARGS && mkdir -p '$RUNS/attributes' && cp '$CACHE/attribute_probe.pt' '$CACHE/ffhq256/attributes.json' '$RUNS/attributes/'"
stage rae_gate $PY scripts/rae_reconstruction_gate.py --data "$DATA/ffhq-256" $GATE_ARGS --out "$RUNS/rae_gate"
TH=$(teacher_hours); echo "[run_all] teacher budget: $TH h"
CFG=configs/pod_teacher_b.yaml
stage teacher $PY scripts/train.py $CFG out_dir="$RUNS" data.cache="$CACHE/ffhq256" budget.max_train_hours="$TH" budget.usd_per_hour="$USD_PER_HOUR" $TEACHER_ARGS
T=$RUNS/teacher_b/ckpt/ema.safetensors
EV="--data $DATA/ffhq-256 --cache $CACHE/ffhq256 --probe $CACHE/attribute_probe.pt --out $RUNS/eval"
stage eval_teacher $PY scripts/evaluate.py --ckpt "$T" --tag teacher $EV $EVAL_ARGS
stage wflow $PY scripts/onestep.py wflow --teacher "$T" --cache "$CACHE/ffhq256" --out "$RUNS/student_wflow" $STUDENT_ARGS
stage gan $PY scripts/onestep.py gan --teacher "$T" --cache "$CACHE/ffhq256" --out "$RUNS/student_gan" $STUDENT_ARGS
for s in wflow gan; do
  stage eval_$s $PY scripts/evaluate.py --ckpt "$RUNS/student_$s/student.safetensors" --kind student --tag ${s}_1step $EV $EVAL_ARGS --steps 1
done
stage package bash -c "cd '$KEEP' && tar czf results.tar.gz --exclude='*.pt' --exclude='latest.pt' --exclude='*_images' runs logs && ls -la results.tar.gz"
echo "[run_all] ALL STAGES COMPLETE ($(date -u +%H:%M:%S))"
if [ "${AUTO_STOP:-0}" = 1 ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  # Give the operator time to copy results (they touch $STATE/pulled), then stop to end GPU billing.
  for _ in $(seq 1 "${GRACE_MIN:-60}"); do [ -f "$STATE/pulled" ] && break; sleep 60; done
  echo "[run_all] stopping pod $RUNPOD_POD_ID (volume kept; terminate after copying results)"
  runpodctl stop pod "$RUNPOD_POD_ID" || true
fi
