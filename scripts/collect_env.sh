#!/usr/bin/env bash
# Snapshot of the hardware and software a result was measured on.
# Writes to stdout; redirect into results/env.txt. Contains no hostnames or IPs.
set -uo pipefail

echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "os: $(. /etc/os-release && echo "$PRETTY_NAME")"
echo "kernel: $(uname -r)"
echo "cpu: $(lscpu | sed -n 's/^Model name: *//p')"
echo "cpus: $(nproc)"
echo "ram_gb: $(awk '/MemTotal/ {printf "%.1f", $2/1048576}' /proc/meminfo)"
echo
echo "## nvidia-smi"
nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm,power.limit --format=csv
nvidia-smi -q | grep -iE 'virtualization mode' || true
echo
echo "## cuda"
nvcc --version 2>/dev/null | tail -1 || echo "nvcc not on host (fine if engines run in containers)"
echo
echo "## container images"
for img in "${VLLM_IMAGE:-vllm/vllm-openai}" "${SGLANG_IMAGE:-lmsysorg/sglang}" \
           "${TRTLLM_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release}"; do
  docker image inspect --format '{{index .RepoDigests 0}}' "$img" 2>/dev/null \
    || echo "$img: not pulled"
done

echo
echo "## engine packages (venv runs; set VENV_DIR, default ~/venvs)"
for v in "${VENV_DIR:-$HOME/venvs}"/*/; do
  [[ -x "$v/bin/python" ]] || continue
  echo "$(basename "$v"): $(uv pip list --python "$v/bin/python" 2>/dev/null \
    | awk 'tolower($1) ~ /^(vllm|sglang|tensorrt-llm|aiperf|torch)$/ {printf "%s==%s ", $1, $2}')"
done
