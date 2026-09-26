#!/usr/bin/env bash
# upload_inputs.sh SESSION [small-only]: copy runs/colab/inputs (scripts/colab/stage_inputs.py) to /content/inputs on
# a Colab VM. Keeps the watchdog hold fresh while it runs. The teacher (731 MB) goes in 40 MB parts -- the transfer
# API sends each file as one base64 request and refuses ~50 MB+ -- and is reassembled + sha256-checked on the VM.
# Needs the Colab CLI (Linux/macOS/WSL): uv tool install google-colab-cli
set -u
S=$1
HERE=$(cd "$(dirname "$0")" && pwd)
IN=$HERE/../../runs/colab/inputs
HOLD=~/.humandit_watch/hold/$S
mkdir -p "$(dirname "$HOLD")"
export PATH="$HOME/.local/bin:$PATH"
up() {  # up LOCAL REMOTE_NAME, 3 tries
  for try in 1 2 3; do
    touch "$HOLD"
    if colab upload -s "$S" "$1" "/content/inputs/$2" 2>&1 | grep -q Uploaded; then return 0; fi
    echo "[$S] $2 attempt $try failed"; sleep 5
  done
  echo "[$S] UPLOAD FAILED: $2"; exit 1
}
touch "$HOLD"
echo "import os; os.makedirs('/content/inputs', exist_ok=True)" | colab exec -s "$S" >/dev/null
up "$IN/humandit-code.tar.gz" humandit-code.tar.gz
up "$IN/fixed_images.npy" fixed_images.npy
[ "${2:-}" = small-only ] && { echo "[$S] small inputs uploaded"; exit 0; }
for f in attribute_probe.pt attributes.json ffhq256_dinov2_real.npz teacher_ema.safetensors.sha256; do up "$IN/$f" "$f"; done
P=$(mktemp -d)
split -b 40M -d "$IN/teacher_ema.safetensors" "$P/teacher_ema.safetensors.part"
t0=$(date +%s)
for f in "$P"/teacher_ema.safetensors.part*; do up "$f" "$(basename "$f")"; done
rm -rf "$P"
echo "[$S] teacher parts uploaded in $(( $(date +%s) - t0 )) s"
touch "$HOLD"
colab exec -s "$S" --timeout 300 -f "$HERE/assemble.py" 2>&1 | tail -2
