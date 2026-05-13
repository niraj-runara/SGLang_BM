#!/usr/bin/env bash
# Start SGLang OpenAI-compatible server for moonshotai/Kimi-K2.6.
#
# Defaults assume eight assignment GPUs: 8x NVIDIA RTX PRO 6000 (Blackwell), tensor-parallel across one server.
# Override TP_SIZE for smoke tests (e.g. TP_SIZE=1 on one card). If prefill/KV OOMs at 32k–128k, tune
# CONTEXT_LENGTH and/or MEM_FRACTION_STATIC, or adjust parallelism per SGLang docs.
#
# Native INT4 / compressed-tensors layouts are usually auto-detected from config.json — avoid
# duplicating --quantization unless Moonshot's deployment guide requires it for your SGLang build.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-K2.6}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-131072}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.88}"
TP_SIZE="${TP_SIZE:-8}"
DP_SIZE="${DP_SIZE:-1}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-256}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
REASONING_PARSER="${REASONING_PARSER:-kimi}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-kimi_k2}"
# Optional explicit quantization (leave empty for auto); examples: compressed-tensors, awq
QUANTIZATION="${QUANTIZATION:-}"
# Extra args appended verbatim (e.g. --kv-cache-dtype fp8_e4m3)
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ ! -x "$(command -v python3)" ]]; then
  echo "python3 not found"
  exit 1
fi

LAUNCH=(python3 -m sglang.launch_server
  --model-path "${MODEL_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --context-length "${CONTEXT_LENGTH}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --tensor-parallel-size "${TP_SIZE}"
  --data-parallel-size "${DP_SIZE}"
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --reasoning-parser "${REASONING_PARSER}"
  --tool-call-parser "${TOOL_CALL_PARSER}"
)

if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  LAUNCH+=(--trust-remote-code)
fi

if [[ -n "${QUANTIZATION}" ]]; then
  LAUNCH+=(--quantization "${QUANTIZATION}")
fi

# shellcheck disable=SC2206
EXTRA_ARR=(${EXTRA_ARGS})
LAUNCH+=("${EXTRA_ARR[@]}")

echo "Launching: ${LAUNCH[*]}"
exec "${LAUNCH[@]}"
