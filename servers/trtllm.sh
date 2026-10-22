#!/usr/bin/env bash
# TensorRT-LLM via trtllm-serve (OpenAI-compatible).
# Flags checked against the trtllm-serve docs on 2026-10-21: --max_seq_len, --config
# (alias --extra_llm_api_options), --tool_parser qwen3. TODO: verify the kv_cache_config keys
# and the image tag on the pinned release. Set ENGINE_VENV to run from a venv.
source "$(dirname "$0")/common.sh"
IMAGE="${TRTLLM_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:latest}"   # TODO: pin a tag in phase 1

extra=$(mktemp)
cat > "$extra" <<YAML
# Chunked prefill is off by default and then a prompt over max_num_tokens (8,192) is rejected with
# HTTP 400; agent prompts grow past that. vLLM and SGLang chunk prefills by default.
enable_chunked_prefill: true
kv_cache_config:
  free_gpu_memory_fraction: $GPU_MEM_FRACTION
  enable_block_reuse: $([[ "$PREFIX_CACHE" == 1 ]] && echo true || echo false)
YAML

args=(--host 0.0.0.0 --port "$PORT" --max_seq_len "$MAX_MODEL_LEN"
  --tool_parser qwen3)   # parser only matters for `apg quality`

# ALIGNED=1: the same explicit scheduler limits on every engine (see docs/progress.md, equal settings)
[[ "${ALIGNED:-0}" == 1 ]] && args+=(--max_batch_size 128 --max_num_tokens 8192)

if [[ -n "${ENGINE_VENV:-}" ]]; then
  exec "$ENGINE_VENV/bin/trtllm-serve" "$MODEL" "${args[@]}" --extra_llm_api_options "$extra"
fi
exec docker run "${DOCKER_COMMON[@]}" -v "$extra:/extra.yaml:ro" "$IMAGE" \
  trtllm-serve "$MODEL" "${args[@]}" --extra_llm_api_options /extra.yaml
