#!/usr/bin/env bash
# vLLM OpenAI-compatible server. Set ENGINE_VENV to run from a venv instead of a container.
source "$(dirname "$0")/common.sh"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"   # TODO: pin a tag in phase 1

args=(--port "$PORT" --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEM_FRACTION"
  --enable-auto-tool-choice --tool-call-parser hermes)   # parser only matters for `apg quality`
# ALIGNED=1: the same explicit scheduler limits on every engine (see docs/progress.md, equal settings)
[[ "${ALIGNED:-0}" == 1 ]] && args+=(--max-num-batched-tokens 8192 --max-num-seqs 128)
[[ "$PREFIX_CACHE" == 1 ]] && args+=(--enable-prefix-caching) || args+=(--no-enable-prefix-caching)

if [[ -n "${ENGINE_VENV:-}" ]]; then
  export VLLM_USE_FLASHINFER_SAMPLER=0   # FlashInfer sampler JIT failed: its log says nvcc or ninja is missing
  exec "$ENGINE_VENV/bin/vllm" serve "$MODEL" "${args[@]}"
fi
exec docker run "${DOCKER_COMMON[@]}" "$IMAGE" --model "$MODEL" "${args[@]}"
