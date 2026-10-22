# PLAN: agents-per-gpu

This is the working plan. Work through the phases in order. Each phase ends with acceptance
criteria, and a phase is done only when every criterion is met and checked by running
something. Read [AGENTS.md](AGENTS.md) first.

Claims about engine features and versions below were written on 2026-10-20 from memory.
Anything marked `TODO: verify` must be checked against current docs before it is relied on.

**Status (2026-10-22): phases 1 to 5 are done and v1.0 is released.** The results are in [reports/](reports/) with the written analysis in
[reports/REPORT.md](reports/REPORT.md); the day-by-day log, deviations from this plan and the open items are in
[docs/progress.md](docs/progress.md). The text below is the plan as written and amended; where it disagrees with
the report, the report and the progress log are right.

## Context

- **Hardware (changed 2026-10-21):** a whole NVIDIA A40 (48 GB, Ampere GA102) rented
  by the hour. The earlier plan used a 16 GB A40-16Q vGPU slice on another VM, which went away.
  Memory is no longer capped to 16 GB. Ampere supports BF16, INT8 and INT4, but not FP8.
- **Starting point:** a working TensorRT-LLM "hello world" on this VM with TinyLlama-1.1B-Chat.
- **Time box:** about 5 weeks while the VM is available. Phases 1–4 are the core.
  Phase 5 is the write-up. The stretch goals are optional.

## Definitions

### Workloads (`configs/workloads.yaml`)

- **agent:** seeded synthetic sessions. Each session has:
  - a shared system prompt with N tool schemas, about 2,000 tokens, identical across sessions;
  - a user task of 50–200 tokens;
  - 3–8 steps. Each step is a model output of 40–150 tokens (a tool call), then a tool
    result of 200–1,500 tokens appended to the context;
  - a final answer of 150–400 tokens.

  A session's next step is sent only after the previous step finishes. That closed loop is
  what makes it an agent workload.
- **chat:** the control workload. Single-turn requests with no shared prefix, and prompt and
  output lengths drawn to give about the same total tokens per request.
- **Output lengths are fixed** per request, using `max_tokens` plus the engine's ignore-EOS
  option. That way every engine and model generates the same number of tokens.
  Verified 2026-10-21/22: all three engines honour `ignore_eos` (every checked request had `length_ok`).

### Metrics (per request, stored as raw JSONL)

- TTFT, inter-token latency (ITL), step latency (send → last token), prompt and output token
  counts, the success flag and the timestamps.
- Per session: task latency (the sum of steps, including queueing).
- Derived metrics: p50/p95/p99, output tok/s, completed sessions per minute.

### SLOs (`configs/slo.yaml`)

These SLOs are choices, not industry standards. The report says so.

| Profile | Condition at p95 |
|---|---|
| interactive | TTFT ≤ 1.0 s and ITL ≤ 50 ms |
| agent-step | step latency ≤ 3.0 s |

**Capacity** is the highest concurrency level, in active agent sessions, at which the p95
values still meet the SLO and fewer than 1% of requests fail. It is measured across 3 repeated
runs, and the report gives the worst of the three.

### Cost

$/hour for an A40 comes from a public cloud price list, with the date and source recorded.
It is a parameter, and `TODO: verify` the price before publishing. The "slice" in the code is now the whole GPU (`slice_fraction` 1.0). From it:

- $/1M output tokens = hourly cost ÷ (output tok/s at capacity × 3600) × 1e6
- $/agent task = hourly cost ÷ (sessions completed per hour at capacity)

## Test matrix (`configs/matrix.yaml`)

| Variant | Weights (approx.) | TRT-LLM | vLLM | SGLang |
|---|---|---|---|---|
| qwen3-4b-bf16 | ~8 GB | ✓ | ✓ | ✓ |
| qwen3-8b-int8 (W8A8 or weight-only) | ~9 GB | ✓ | ✓ | ✓ |
| qwen3-8b-int4-awq | ~6 GB | ✓ | ✓ | ✓ |
| qwen3-8b-bf16 | ~16 GB | ✓ | ✓ | ✓ (fits on 48 GB; the quality reference) |

- Qwen3 is used because the 4B and 8B models share a tokenizer and a chat template, support
  tool calling and are Apache-2.0. Use non-thinking mode (`enable_thinking=False`).
  Verified: all three engines serve Qwen3-4B and Qwen3-8B, so the Llama fallback was not needed.
- Use the same quantized checkpoint on every engine where possible, for example the official
  `Qwen/Qwen3-8B-AWQ`. Where an engine needs its own quantization path, such as TensorRT Model
  Optimizer for TRT-LLM, use that path and record the difference in the results.
  Verified: vLLM and SGLang load all variants; TensorRT-LLM 1.2.1 loads only BF16 from the published
  checkpoints (see the matrix in `reports/results.md` and REPORT.md for the two failures and the
  Model Optimizer workaround).
- An empty cell is a finding, not a failure. Record why it couldn't run.
- Other axes: workload {agent, chat} × prefix caching {on, off} ×
  concurrency {1, 2, 4, 8, 16, 24, 32, 48, 64, 96}. The sweep stops early once the SLO breaks
  by more than 2×.

## Phase 1: Environment and a trustworthy client (week 1)

1. `scripts/collect_env.sh`: GPU, vGPU profile, driver, CUDA, container image digests, CPU,
   RAM, and the date.
2. `servers/*.sh`: launch Qwen3-4B BF16 on each engine from a pinned container image, with an
   OpenAI-compatible endpoint on port 8000. Run one engine at a time.
3. `apg` client: async and streaming, it records TTFT, ITL and step latency per request. Its
   timing comes from the client's monotonic clock.
4. Cross-check the client against NVIDIA AIPerf (the successor to GenAI-Perf) on one
   fixed-length config. Verified: `pip install aiperf`, flags as in `scripts/crosscheck_aiperf.sh`.

**Acceptance:**
- All three engines answer a streamed chat request for Qwen3-4B.
- At concurrency 1, 8 and 32, the client's TTFT and output tok/s are within 5% of AIPerf's
  on the same config, or the difference is explained in `docs/progress.md`.
- `results/env.txt` is committed.

## Phase 2: Workloads and sweeps (week 2)

1. `apg workload`: generates the agent and chat workloads from a seed into `workloads/`,
   using the model's own tokenizer, so token counts are exact.
2. `apg sweep`: runs the concurrency ladder. It warms up first, then measures for a fixed
   duration or session count, and repeats the run 3 times.
3. Prefix caching on and off, on each engine, for both workloads.
4. Measure noise: run the same config 5 times on different days and report the
   coefficient of variation. A rented host can still be shared (CPU, PCIe, neighbours), so this stays.

**Acceptance:**
- Raw JSONL is saved for every run, and the sweep is reproducible from one command.
- There is a first plot: step latency p95 against concurrency, agent and chat, for each
  engine, for Qwen3-4B BF16.
- The noise level is known and written down. If it's above 10%, raise the number of repeats.

## Phase 3: Quantization and quality (week 3)

1. Serve the Qwen3-8B INT8 and INT4-AWQ variants on each engine. Record the quantization path
   and checkpoint used for each.
2. Quality evaluation through the same OpenAI endpoint. Use a fixed subset of the Berkeley
   Function Calling Leaderboard (BFCL), simple + multiple categories, about 200 cases, scored
   on whether the tool call is correct. Verified: BFCL is Apache-2.0 (see `docs/licenses.md`).
3. Reference: Qwen3-8B BF16 fits on the A40, so run it through the same cases as the reference.

**Acceptance:**
- There is an accuracy table for each variant and engine.
- Quantized variants that lose more than 3 points against the reference are flagged.
- Every cell in the matrix has a capacity number or a documented reason for being empty.

## Phase 4: Capacity, cost and the planner (week 4)

1. `apg report`: capacity tables, latency curves, and a Pareto plot of accuracy against agents
   per GPU.
2. `apg plan --users N --sessions-per-user R --slo <profile>`: from the measured capacity, it
   outputs the best engine and variant, the GPUs needed, $/hour, $/1M tokens and $/agent task.
   Its assumptions are printed.
3. Unit tests for the metric maths, the SLO check and the planner.

**Acceptance:**
- `apg report` regenerates every table and plot in `reports/` from `results/` alone.
- `apg plan` gives the same answer as a hand calculation on one documented example.
- `pytest` and `ruff` pass.

## Phase 5: Write-up and release (week 5)

1. `reports/REPORT.md`, about 1,500 words: the question, setup, method, results, why the
   curves look as they do, limitations, and what would change on H100 or Blackwell
   (as a hypothesis, labelled as one).
2. README: replace the status banner with the headline result and one plot.
3. Release checklist, below.

**Release checklist:**
- [x] A fresh clone plus the documented commands reproduces one result (checked 2026-10-22: `apg report` regenerates
  `reports/results.md` and `runs.csv` byte for byte, apart from the generation date).
- [x] Every number in the README and report traces to a file in `results/` (checked against `reports/results.md`).
- [ ] Container image digests: **not met**, the engines ran from venvs. Engine versions and the environment
  snapshot (`results/env.txt`) are committed instead.
- [x] No credentials, hostnames, IPs or VM provider details in the repo or its history (the history was squashed
  to one commit and scanned on 2026-10-22).
- [x] The limitations section is complete.
- [x] A tagged release `v1.0`.

## Stretch goals (only after phase 5)

- **Speculative decoding:** Qwen3-8B with a small draft model or EAGLE. Measure the capacity
  change at the SLO, not only single-stream speed.
- **NVIDIA NIM:** the same sweep against a NIM container for one variant.
- **A Nemotron model:** for example a Nemotron Nano model, as an extra row in the matrix.
  Not attempted.
- **Chunked prefill:** test it on and off, since agent steps send long prompts.
- **Nsight Systems** traces of one engine at capacity, if the rented host allows profiling.
  Not attempted.

## Changes made while implementing (2026-10-21)

See `docs/progress.md` for the reasons.

- **Verified:** AIPerf's name, install (`pip install aiperf`) and flags; BFCL's licence (Apache-2.0)
  and data format; the Qwen3 licences; the `trtllm-serve` flags used in `servers/trtllm.sh`.
  Resolved later: ignore-EOS per engine, the AIPerf JSON export layout (`{tag: {unit, avg}}`), and the
  tool-call parser names (vLLM `hermes`, SGLang `qwen25`; both worked in the quality runs). Image tags do not
  apply: the engines ran from venvs.
- **Workloads:** a third profile, `fixed` (1,000 in, 128 out), is used for the AIPerf cross-check.
  Session context holds the workload's synthetic tool calls, not the model's replies, so prompts
  are identical across engines.
- **Warm-up:** ends after `max(warmup_requests, concurrency)` completions. Warm-up and drain requests
  stay in the raw file, tagged, and are excluded from summaries.
- **Prefix caching:** a per-run nonce after the shared prefix keeps repeated runs from hitting the
  cache on session-specific text. The shared system prompt stays a common prefix.
- **Capacity:** per repeat, the last passing level before the first failure. Reported value: the
  minimum across repeats. A `+` marks a ladder that never broke the SLO.
- **Quality:** 100 BFCL `simple_python` and 100 `multiple` cases, seed 0, simplified checker. Not
  comparable with the BFCL leaderboard. The server's tool-call parser has to be on.
- **Planner:** `--users N --sessions-per-user R` means R concurrent agent sessions per user. The
  worked example is in `docs/planner.md`.
- **Empty matrix cells** are recorded with `apg cell` and shown in the report.

## Risks

| Risk | Mitigation |
|---|---|
| Noise from other VMs on the shared GPU | Repeats, CV reporting, worst-of-3 capacity, and runs at quiet hours |
| An engine doesn't support a variant | Record it as a finding, and use the fallback model family if it hits many cells |
| The VM goes away mid-project | Commit raw results after every session. Reports regenerate from `results/` alone. |
| Engine versions change mid-project | Pin image digests, and don't upgrade until phase 5 |
