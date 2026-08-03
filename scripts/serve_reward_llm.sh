#!/usr/bin/env bash
# Serve frozen Qwen for Falsifier + Adjudicator (REWARD_BACKEND=http).
# Prefer a second GPU: CUDA_VISIBLE_DEVICES=1 ./scripts/serve_reward_llm.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-$ROOT/models}"
PORT="${REWARD_PORT:-8001}"
GPU_MEM="${REWARD_GPU_MEM:-0.90}"
HOST="${REWARD_HOST:-0.0.0.0}"

export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

echo "Serving reward LLM from $MODEL_PATH on ${HOST}:${PORT}"
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEM" \
  --max-model-len 4096 \
  --trust-remote-code
