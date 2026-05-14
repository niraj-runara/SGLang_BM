#!/usr/bin/env bash
# Download moonshotai/Kimi-K2.6 weights separately from serving.
#
# Defaults to a repo-local model directory so launch_sglang_kimi.sh can serve
# from disk without implicitly downloading from Hugging Face.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ID="${MODEL_ID:-moonshotai/Kimi-K2.6}"
MODEL_DIR="${MODEL_DIR:-${ROOT}/models/Kimi-K2.6}"
REVISION="${REVISION:-}"

if [[ ! -x "$(command -v hf)" ]]; then
  echo "hf (Hugging Face CLI) not found. Activate the venv or run scripts/setup_gpu_host.sh first."
  exit 1
fi

mkdir -p "${MODEL_DIR}"

DOWNLOAD=(hf download "${MODEL_ID}" --local-dir "${MODEL_DIR}")
if [[ -n "${REVISION}" ]]; then
  DOWNLOAD+=(--revision "${REVISION}")
fi

echo "Downloading ${MODEL_ID} to ${MODEL_DIR}"
echo "Command: ${DOWNLOAD[*]}"
"${DOWNLOAD[@]}"

echo ""
echo "Done. Serve from the local copy with:"
echo "  MODEL_PATH=${MODEL_DIR} bash scripts/launch_sglang_kimi.sh"
