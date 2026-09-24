#!/usr/bin/env bash
# Pod environment: uv + Python 3.14 + torch 2.14 wheel matched to the host driver.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
drv=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+' | head -1)
if [ "${drv:-0}" -lt 13 ]; then
  echo "[setup] host driver supports CUDA ${drv}.x < 13 -> using cu126 torch wheels"
  sed -i 's#download.pytorch.org/whl/cu130#download.pytorch.org/whl/cu126#; s#pytorch-cu130#pytorch-cu126#g' pyproject.toml
  uv lock -q
fi
uv sync --extra eval -q
uv run --no-sync python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable: torch wheel does not match the host driver"
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
for _ in range(3):
    x @ x
torch.cuda.synchronize()
import time
t = time.perf_counter()
for _ in range(20):
    x @ x
torch.cuda.synchronize()
tflops = 2 * 4096**3 * 20 / (time.perf_counter() - t) / 1e12
print(f"[setup] torch {torch.__version__} cuda {torch.version.cuda} on {torch.cuda.get_device_name()}: bf16 GEMM {tflops:.0f} TFLOPS")
PY
