#!/usr/bin/env bash
# One-shot environment prep for SGLang + Kimi-K2.6 benchmarking.
#
# Assignment target hardware (per brief): 8x NVIDIA RTX PRO 6000 (Blackwell), 4-bit native quant weights.
# SGLang serves one sharded replica with TP_SIZE=8 by default (see launch_sglang_kimi.sh).
# Tested layout: Ubuntu 22.04/24.04, Python 3.10–3.12, proprietary NVIDIA driver (nvidia-smi works).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-${ROOT}/.venv}"
# RTX PRO 6000 Blackwell benefits from a recent CUDA userland; override if your image pins another wheel.
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
SKIP_TORCH="${SKIP_TORCH:-0}"
SKIP_SGLANG="${SKIP_SGLANG:-0}"

echo "[1/5] Checking GPU driver..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. Install the NVIDIA driver before continuing."
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo "[2/5] Creating venv at ${VENV}..."
python3 -m venv "${VENV}"
# shellcheck source=/dev/null
source "${VENV}/bin/activate"
python -m pip install -U pip setuptools wheel

if [[ "${SKIP_TORCH}" != "1" ]]; then
  echo "[3/5] Installing PyTorch (CUDA wheels from ${TORCH_INDEX_URL})..."
  pip install --index-url "${TORCH_INDEX_URL}" "torch" "torchvision" "torchaudio"
else
  echo "[3/5] Skipping PyTorch install (SKIP_TORCH=1)."
fi

if [[ "${SKIP_SGLANG}" != "1" ]]; then
  echo "[4/5] Installing SGLang..."
  if ! pip install "sglang[all]"; then
    echo "sglang[all] failed; retrying with base sglang (install optional extras manually if needed)."
    pip install "sglang"
  fi
else
  echo "[4/5] Skipping sglang install (SKIP_SGLANG=1)."
fi

echo "[5/5] Installing benchmark client libraries..."
pip install -r "${ROOT}/requirements-benchmark.txt"

echo "Done. Activate with:"
echo "  source ${VENV}/bin/activate"
echo ""
echo "If Kimi weights are gated, export a token before launch:"
echo "  export HF_TOKEN=..."
