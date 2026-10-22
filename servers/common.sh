#!/usr/bin/env bash
# Shared settings for the engine launch scripts. Source it; don't run it.
# Image tags are pinned once in phase 1; record digests in results/env.txt.
set -euo pipefail

VARIANT="${1:?usage: $0 <variant>   (see configs/models.yaml)}"
case "$VARIANT" in
  qwen3-4b-bf16)     MODEL=Qwen/Qwen3-4B ;;
  qwen3-8b-bf16)     MODEL=Qwen/Qwen3-8B ;;
  qwen3-8b-int8)     MODEL=JunHowie/Qwen3-8B-GPTQ-Int8 ;;   # community GPTQ weight-only INT8
  qwen3-8b-int4-awq) MODEL=Qwen/Qwen3-8B-AWQ ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 1 ;;
esac

MODEL="${MODEL_OVERRIDE:-$MODEL}"   # e.g. a locally quantized checkpoint (record it with --quant-path)
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEM_FRACTION="${GPU_MEM_FRACTION:-0.90}"
PREFIX_CACHE="${PREFIX_CACHE:-1}"              # 1 = on, 0 = off
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

DOCKER_COMMON=(--rm --gpus all --ipc=host -p "$PORT:$PORT"
  -v "$HF_CACHE:/root/.cache/huggingface" -e HF_TOKEN)
