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
uv run python -c "import torch; print('[setup] torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(), 'bf16', torch.cuda.is_bf16_supported())"
