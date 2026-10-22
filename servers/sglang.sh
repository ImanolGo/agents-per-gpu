#!/usr/bin/env bash
# SGLang OpenAI-compatible server. RadixAttention prefix caching is on by default.
# Set ENGINE_VENV to run from a venv instead of a container.
source "$(dirname "$0")/common.sh"
IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:latest}"   # TODO: pin a tag in phase 1

args=(--model-path "$MODEL" --host 0.0.0.0 --port "$PORT"
  --context-length "$MAX_MODEL_LEN" --mem-fraction-static "$GPU_MEM_FRACTION"
  --tool-call-parser qwen25)   # parser only matters for `apg quality`
[[ "$PREFIX_CACHE" == 1 ]] || args+=(--disable-radix-cache)
# ALIGNED=1: the same explicit scheduler limits on every engine (see docs/progress.md, equal settings)
[[ "${ALIGNED:-0}" == 1 ]] && args+=(--chunked-prefill-size 8192 --max-running-requests 128)

if [[ -n "${ENGINE_VENV:-}" ]]; then
  exec "$ENGINE_VENV/bin/python" -m sglang.launch_server "${args[@]}"
fi
exec docker run "${DOCKER_COMMON[@]}" --shm-size 16g "$IMAGE" \
  python3 -m sglang.launch_server "${args[@]}"
