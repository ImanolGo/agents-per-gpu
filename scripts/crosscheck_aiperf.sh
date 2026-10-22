#!/usr/bin/env bash
# Phase 1 cross-check: run the same fixed-length config through apg and AIPerf, then compare.
#   scripts/crosscheck_aiperf.sh <engine> <variant> [concurrency ...]   (default: 1 8 32)
# Run it with the server's prefix caching OFF (PREFIX_CACHE=0 servers/<engine>.sh, APG_PREFIX_CACHE=off):
# AIPerf's synthetic prompts repeat and hit the cache (61% at c=32 on vLLM), apg's do not.
# Needs: a server already up on $URL, `uv run apg workload --profile fixed`, and `pip install aiperf`
# in a separate venv (AIPERF=path/to/aiperf). Writes results/aiperf/ and prints the comparison.
set -euo pipefail

engine="${1:?usage: $0 <engine> <variant> [concurrency ...]}"
variant="${2:?usage: $0 <engine> <variant> [concurrency ...]}"
shift 2
levels=("$@")
[[ ${#levels[@]} -gt 0 ]] || levels=(1 8 32)

URL="${URL:-http://localhost:8000}"
AIPERF="${AIPERF:-aiperf}"
SECONDS_PER_RUN="${SECONDS_PER_RUN:-60}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3-8B}"
model=$(curl -sf "$URL/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')

for c in "${levels[@]}"; do
  out="results/aiperf/$engine/$variant/c$(printf '%03d' "$c")"
  uv run apg run --engine "$engine" --variant "$variant" --workload fixed \
    --concurrency "$c" --measure-seconds "$SECONDS_PER_RUN" --url "$URL" \
    --prefix-cache "${APG_PREFIX_CACHE:-on}" \
    ${ENGINE_VERSION:+--engine-version "$ENGINE_VERSION"}
  "$AIPERF" profile --model "$model" --url "$URL" --endpoint-type chat --streaming \
    --tokenizer "$TOKENIZER" --isl 1000 --osl 128 --extra-inputs ignore_eos:true \
    --use-server-token-count --concurrency "$c" --benchmark-duration "$SECONDS_PER_RUN" \
    --warmup-request-count "$((c > 20 ? c : 20))" --random-seed 0 --ui none \
    --output-artifact-dir "$out"
  echo "== c=$c =="
  uv run apg crosscheck \
    --run "results/raw/$engine/$variant/fixed-pc${APG_PREFIX_CACHE:-on}-c$(printf '%03d' "$c")-r1.jsonl" \
    --aiperf "$out/profile_export_aiperf.json"
done
