#!/usr/bin/env bash
# One-shot environment prep for SGLang + Kimi-K2.6 benchmarking.
#
# Assignment target hardware (per brief): 8x NVIDIA RTX PRO 6000 (Blackwell), 4-bit native quant weights.
# SGLang serves one sharded replica with TP_SIZE=8 by default (see launch_sglang_kimi.sh).
# Tested layout: Ubuntu 22.04/24.04, Python 3.10–3.12, proprietary NVIDIA driver (nvidia-smi works).
#
# FlashInfer (pulled in by SGLang) JIT-compiles some CUDA extensions and needs <curand.h>. Images that only
# ship the driver + partial /usr/local/cuda often lack it. This script checks and, on Ubuntu as root, can
# install cuda-curand-dev-* from NVIDIA's apt repo. Set SKIP_CUDA_DEV_CHECK=1 to skip. Set
# INSTALL_CUDA_CURAND_DEV=0 to refuse automatic apt installs even as root (you must install headers yourself).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-${ROOT}/.venv}"
# RTX PRO 6000 Blackwell: default cu130 so torch/torchvision/torchaudio match PyTorch 2.11 wheels SGLang pulls.
# Use TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 (or cu126) on older driver stacks.
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
SKIP_TORCH="${SKIP_TORCH:-0}"
SKIP_SGLANG="${SKIP_SGLANG:-0}"
SKIP_CUDA_DEV_CHECK="${SKIP_CUDA_DEV_CHECK:-0}"
# Default: allow apt install of cuRAND dev headers on Ubuntu when running as root; set to 0 to disable.
INSTALL_CUDA_CURAND_DEV="${INSTALL_CUDA_CURAND_DEV:-1}"

cuda_pkg_ver_from_torch_index() {
  case "${TORCH_INDEX_URL}" in
    *cu131*) echo "13-1" ;;
    *cu130*) echo "13-0" ;;
    *cu128*) echo "12-8" ;;
    *cu126*) echo "12-6" ;;
    *cu124*) echo "12-4" ;;
    *) echo "13-0" ;;
  esac
}

cuda_pkg_ver_from_nvcc() {
  local nvcc_bin="$1"
  "${nvcc_bin}" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1-\2/p' | head -n1
}

nvidia_cuda_apt_repo_id() {
  # Maps Ubuntu VERSION_ID to NVIDIA cuda repo directory name (x86_64 only here).
  case "${VERSION_ID:-}" in
    22.04) echo "ubuntu2204" ;;
    24.04) echo "ubuntu2404" ;;
    *) return 1 ;;
  esac
}

ensure_cuda_curand_dev_headers() {
  local cuda_home="${CUDA_HOME:-/usr/local/cuda}"
  local curand_h="${cuda_home}/include/curand.h"
  if [[ -f "${curand_h}" ]]; then
    echo "CUDA cuRAND dev header present: ${curand_h}"
    return 0
  fi

  local nvcc_bin=""
  if command -v nvcc >/dev/null 2>&1; then
    nvcc_bin="$(command -v nvcc)"
  elif [[ -x "${cuda_home}/bin/nvcc" ]]; then
    nvcc_bin="${cuda_home}/bin/nvcc"
  fi

  if [[ -z "${nvcc_bin}" ]]; then
    echo "Note: nvcc not on PATH and not under ${cuda_home}/bin; skipping curand.h check."
    echo "      If SGLang later fails JIT-compiling FlashInfer kernels, install a full CUDA toolkit (devel)."
    return 0
  fi

  echo "Detected nvcc at ${nvcc_bin} but missing ${curand_h}."
  echo "FlashInfer (used by SGLang) needs cuRAND C headers at JIT time."

  if [[ "${SKIP_CUDA_DEV_CHECK}" == "1" ]]; then
    echo "SKIP_CUDA_DEV_CHECK=1: continuing without cuRAND headers."
    return 0
  fi

  local pkg_ver
  pkg_ver="$(cuda_pkg_ver_from_nvcc "${nvcc_bin}")"
  if [[ -z "${pkg_ver}" ]] || [[ "${pkg_ver}" != *-* ]]; then
    pkg_ver="$(cuda_pkg_ver_from_torch_index)"
    echo "Could not parse CUDA version from nvcc; using TORCH_INDEX_URL mapping -> cuda-curand-dev-${pkg_ver}"
  else
    echo "Using nvcc CUDA release -> cuda-curand-dev-${pkg_ver}"
  fi

  if [[ -f /etc/os-release ]]; then
    # shellcheck source=/dev/null
    . /etc/os-release
  fi

  if [[ "${ID:-}" == "ubuntu" ]] && [[ "${EUID:-$(id -u)}" -eq 0 ]] && [[ "${INSTALL_CUDA_CURAND_DEV}" == "1" ]]; then
    local repo_id uname_m keyring_deb keyring_url
    if ! repo_id="$(nvidia_cuda_apt_repo_id)"; then
      echo "Automatic cuRAND install is only wired for Ubuntu 22.04 / 24.04 (got VERSION_ID=${VERSION_ID:-unknown})."
    else
      uname_m="$(uname -m)"
      if [[ "${uname_m}" != "x86_64" ]]; then
        echo "Automatic NVIDIA apt repo install is only wired for x86_64 (got ${uname_m})."
      else
        keyring_deb="cuda-keyring_1.1-1_all.deb"
        keyring_url="https://developer.download.nvidia.com/compute/cuda/repos/${repo_id}/x86_64/${keyring_deb}"
        echo "Attempting: apt install cuda-curand-dev-${pkg_ver} (NVIDIA CUDA apt repo + keyring)..."
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -y
        if ! command -v wget >/dev/null 2>&1; then
          apt-get install -y wget
        fi
        if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
          rm -f "/tmp/${keyring_deb}"
          wget -qO "/tmp/${keyring_deb}" "${keyring_url}"
          dpkg -i "/tmp/${keyring_deb}" || apt-get install -f -y
        fi
        apt-get update -y
        if apt-get install -y "cuda-curand-dev-${pkg_ver}"; then
          if [[ -f "${curand_h}" ]]; then
            echo "Installed cuRAND headers: ${curand_h}"
            return 0
          fi
        fi
        echo "Automatic install failed (wrong CUDA repo line for this host, or package cuda-curand-dev-${pkg_ver} missing)."
      fi
    fi
  elif [[ "${INSTALL_CUDA_CURAND_DEV}" == "1" ]] && [[ "${ID:-}" == "ubuntu" ]] && [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    echo "Re-run this step as root to auto-install, for example:"
    echo "  sudo env TORCH_INDEX_URL='${TORCH_INDEX_URL}' bash ${ROOT}/scripts/setup_gpu_host.sh"
    echo "or install matching cuda-curand-dev-* from https://developer.nvidia.com/cuda-downloads"
  fi

  echo "Fix manually, then re-run this script. Example (adjust CUDA version to match nvcc):"
  echo "  sudo apt-get install -y cuda-curand-dev-${pkg_ver}"
  exit 1
}

echo "[1/7] Checking GPU driver..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. Install the NVIDIA driver before continuing."
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo "[2/7] CUDA toolkit headers (cuRAND for FlashInfer JIT)..."
ensure_cuda_curand_dev_headers

echo "[3/7] Creating venv at ${VENV}..."
python3 -m venv "${VENV}"
# shellcheck source=/dev/null
source "${VENV}/bin/activate"
python -m pip install -U pip setuptools wheel

if [[ "${SKIP_TORCH}" != "1" ]]; then
  echo "[4/7] Installing PyTorch (CUDA wheels from ${TORCH_INDEX_URL})..."
  pip install --index-url "${TORCH_INDEX_URL}" "torch" "torchvision" "torchaudio"
else
  echo "[4/7] Skipping PyTorch install (SKIP_TORCH=1)."
fi

if [[ "${SKIP_SGLANG}" != "1" ]]; then
  echo "[5/7] Installing SGLang (>=0.5.10.post1, required by Kimi-K2.6)..."
  if ! pip install "sglang[all]>=0.5.10.post1"; then
    echo "sglang[all] failed; retrying with base sglang (install optional extras manually if needed)."
    pip install "sglang>=0.5.10.post1"
  fi
else
  echo "[5/7] Skipping sglang install (SKIP_SGLANG=1)."
fi

# SGLang can replace torch from PyPI (different CUDA major) while torchvision stays on the pre-sglang wheel.
if [[ "${SKIP_TORCH}" != "1" ]] && [[ "${SKIP_SGLANG}" != "1" ]]; then
  echo "[6/7] Re-aligning torch, torchvision, torchaudio from ${TORCH_INDEX_URL}..."
  pip install --index-url "${TORCH_INDEX_URL}" "torch" "torchvision" "torchaudio"
elif [[ "${SKIP_TORCH}" == "1" ]] || [[ "${SKIP_SGLANG}" == "1" ]]; then
  echo "[6/7] Skipping torch re-align (SKIP_TORCH=${SKIP_TORCH}, SKIP_SGLANG=${SKIP_SGLANG})."
fi

echo "[7/7] Installing benchmark client libraries..."
pip install -r "${ROOT}/requirements-benchmark.txt"
# requirements-benchmark pins HF libs for co-install with SGLang; re-assert if another dep drifted them.
if [[ "${SKIP_SGLANG}" != "1" ]]; then
  pip install "transformers==5.6.0" "huggingface-hub>=1.10.0"
fi

echo "Done. Activate with:"
echo "  source ${VENV}/bin/activate"
echo ""
echo "If Kimi weights are gated, export a token before launch:"
echo "  export HF_TOKEN=..."
