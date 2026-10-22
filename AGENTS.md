# AGENTS.md: rules for working on agents-per-gpu

Read this and [PLAN.md](PLAN.md) before writing code.

## How to work

- Follow PLAN.md phase by phase. Don't start a phase until the previous phase's acceptance
  criteria pass.
- At the end of each work session, add a short entry to `docs/progress.md`: what was done,
  what was measured, and what is still open.
- When the plan and reality disagree, for example because an engine flag changed or a variant
  doesn't load, write down the change and the reason in `docs/progress.md` and update
  PLAN.md. Don't quietly drift.
- Engine CLIs and flags change between releases. Check the current docs for TensorRT-LLM,
  vLLM and SGLang instead of relying on memory. Resolve `TODO: verify` markers by checking,
  and record the source and date.

## Honesty about results

- Every number in README.md or `reports/` must be regenerable from a file in `results/` by
  `apg report`. Don't type numbers in by hand.
- Each result records the engine, the container image digest, the model variant, the
  quantization path, the workload seed, the concurrency, the SLO profile and the date.
- Report capacity as the worst of the repeated runs, and say how many runs there were.
- An engine × variant cell that can't run is a result. Record the error and the reason. Never
  drop it silently.
- Don't compare engines with different settings without saying so. That includes max batch
  size, KV cache fraction and prefix caching.
- Label hypotheses as hypotheses, for example what would change on H100.

## Fair benchmarking

- Run one engine at a time on the GPU. Check with `nvidia-smi` that nothing else of ours is
  running before a sweep.
- Use the same workload files, client, tokenizer and output lengths for every engine.
- Warm up before measuring, and discard the warm-up requests.
- Give each engine its documented, recommended serving settings, and tune it equally. If one
  engine is tuned, the others get the same effort, and the report says so.

## Public repo hygiene

- This repo is public. Never commit credentials, Hugging Face tokens, hostnames, IPs, or
  details of the VM provider or account.
- Only public models and public datasets, with their licences recorded in `docs/`.
- No content from private or employer projects.

## Code standards

- Python 3.12, managed with `uv`. The package is `agents_per_gpu`, and the CLI is `apg`.
- `uv run ruff check`, `uv run ruff format --check` and `uv run pytest` must pass before a
  phase is closed.
- Keep dependencies small: an async HTTP client, numpy/pandas, matplotlib, pyyaml and a
  tokenizer library.
- Configuration lives in `configs/*.yaml`, not in code. One CLI command does one step.
- Raw results are append-only JSONL. Never edit them by hand. Rerun instead.

## Running on the VM

```
scripts/collect_env.sh > results/env.txt
servers/<engine>.sh <variant>          # one engine at a time, on port 8000
uv run apg sweep --engine <engine> --variant <variant> --workload agent
uv run apg report
```
