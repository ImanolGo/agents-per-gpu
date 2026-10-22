# agents-per-gpu

**How many AI agents can one GPU serve before they feel slow, and what does each agent task cost?**

![Dumbbell chart: concurrent Qwen3-4B agent sessions one NVIDIA A40 serves within the interactive SLO, with prefix caching off and on. vLLM 2 to 6, SGLang 3 to 6, TensorRT-LLM 2 to 6. Cost per agent task falls from 0.0018 to 0.0006 USD on vLLM, 0.0013 to 0.0005 on SGLang and 0.0018 to 0.0006 on TensorRT-LLM.](reports/hero_prefix_caching.png)

## In plain words

An AI agent doesn't ask a model one question. It works in a loop: read the task, call a tool, read the
result, decide the next step. Every step resends the same long instructions and tool descriptions (about
2,000 tokens) plus everything that has happened so far, and the user waits for the whole chain. Most LLM
serving benchmarks measure chat traffic instead, so they say little about how many agents a GPU can run.

This repo measures that. It replays seeded, agent-shaped sessions against three open-source serving
engines, **vLLM**, **SGLang** and **NVIDIA TensorRT-LLM**, on one rented NVIDIA A40 (48 GB). It raises the
number of concurrent agents step by step and records the last level at which latency still meets a target
(an SLO). That number, **agents per GPU**, and the **cost per agent task** at that load are the results.

## What I found

- **Prefix caching is the biggest lever.** Reusing the cached shared prefix lets every engine serve 6
  concurrent Qwen3-4B agents within the `interactive` SLO, against 2 to 3 without it, and cuts the cost per
  task to between a third and two fifths.
- **The engines are close at their default settings.** All three serve 6 agents at the `interactive` SLO.
  With an 8 s budget per agent step, SGLang leads with 4, then vLLM with 3 and TensorRT-LLM with 2.
- **A quantized bigger model beats a smaller full-precision one.** Qwen3-8B in INT4-AWQ serves 4 to 6
  agents at the 8 s step SLO, against 2 to 4 for Qwen3-4B in BF16, at about the same tool-calling accuracy
  (91 to 92 %). Qwen3-8B in BF16 is too slow per token to meet that budget at all.

![Grouped bar chart: agents per A40 at p95 step latency of 8 s or less, by model and engine (vLLM, SGLang, TensorRT-LLM). Qwen3-4B BF16: 3, 4, 2, accuracy 91.0 %. Qwen3-8B BF16: 0, 0, 0, accuracy 92.0 to 93.0 %. Qwen3-8B INT8: 2, 2, failed to load, accuracy 91.5 to 93.0 %. Qwen3-8B INT4-AWQ: 4, 6, 4 (TensorRT-LLM with its own checkpoint), accuracy 91.0 to 92.0 %.](reports/model_size.png)

Measured 2026-10-21 and 22 on one whole A40. This is one GPU, a synthetic workload, SLOs I chose and
default engine settings, so read it as a snapshot (see [Limitations](#limitations)). Every number and both
graphics regenerate from `results/` with `apg report`. The full analysis is
[reports/REPORT.md](reports/REPORT.md).

## Why agent traffic is different

- **Long shared prefix.** Every request repeats the same system prompt and tool schemas,
  which is often 2,000 tokens or more.
- **Growing context.** Each step appends the previous tool call and its result, so the
  prompt grows turn by turn.
- **Short outputs, many round trips.** Most steps produce a short tool call. The user waits
  for the whole chain, not for one answer.

So the headline here is not peak tokens per second. It is the **number of concurrent agents served while the
latency SLO still holds**, plus the **cost per agent task** at that load. Every engine gets the same workload
and the same client.

## The questions

1. **Engines.** On agent traffic, which engine serves the most concurrent agents within the
   SLO? Does the ranking change from chat traffic?
2. **Big and quantized, or small at full precision?** Qwen3-8B in BF16 fits on 48 GB, so it
   is the quality reference. Is Qwen3-8B in INT4 or INT8 a better deal than the BF16 models,
   on both capacity *and* tool-calling accuracy?
3. **Prefix caching.** How much capacity does KV-cache reuse of the shared prefix buy on agent
   traffic, compared with chat traffic?
4. **Sizing.** Given a number of users and an SLO, how many GPUs do you need, and what does it
   cost? The `apg plan` command answers this from the measured data.

## What's different here

- **Agent-shaped workload.** A seeded generator produces multi-step agent sessions with a
  shared tool-schema prefix, a growing context and short tool-call outputs. A chat workload
  runs alongside as the control.
- **Goodput at an SLO, not peak throughput.** Concurrency is increased until p95 latency
  breaks the SLO. The last concurrency level that passes is the capacity.
- **Speed has to keep quality.** A quantized variant only counts if its tool-calling accuracy
  holds on a fixed evaluation set.
- **Honest about the hardware.** The GPU is a rented, whole A40 on a shared host. Run-to-run variance is measured and reported, not hidden.
  A first run on a time-sliced A40-16Q vGPU exists in the history; it is a different setup and is not mixed with these results.

## Setup

| | |
|---|---|
| GPU | NVIDIA A40 (whole GPU, 48 GB, Ampere GA102, no FP8) |
| Driver / CUDA | 595.99, CUDA 13.0 toolkit (snapshot in [results/env.txt](results/env.txt)) |
| Engines | vLLM 0.32.0, SGLang 0.5.9, TensorRT-LLM 1.2.1 (PyTorch backend). All are served through their OpenAI-compatible endpoints, from venvs (no container digests). |
| Models | Qwen3-4B (BF16); Qwen3-8B (BF16, INT8, INT4-AWQ) |
| Client | `apg`: an async streaming client in this repo, cross-checked against NVIDIA AIPerf |

The full test matrix, the SLO definitions and the methodology are in [PLAN.md](PLAN.md).

## Results

Capacity is the highest concurrency, in active agent sessions, before the first SLO failure, taken as the worst of
the repeated runs. Qwen3-4B BF16, agent workload, prefix caching on (worst of 6 runs over 2 days):

| engine | `interactive` (TTFT p95 <= 1 s, ITL p95 <= 50 ms) | step p95 <= 8 s | USD per agent task at `interactive` capacity |
|---|---|---|---|
| vLLM | 6 | 3 | 0.0006 |
| SGLang | 6 | 4 | 0.0005 |
| TensorRT-LLM | 6 | 2 | 0.0006 |

Model size and quantization, agent workload, prefix caching on. Capacity is at the 8 s step SLO; accuracy is on a
fixed 200-case tool-calling subset (a simplified BFCL-style checker, not leaderboard scores; one point is two cases):

| variant | capacity vLLM / SGLang / TensorRT-LLM | accuracy % vLLM / SGLang / TensorRT-LLM |
|---|---|---|
| Qwen3-4B BF16 | 3 / 4 / 2 | 91.0 / 91.0 / 91.0 |
| Qwen3-8B BF16 | 0 / 0 / 0 | 92.0 / 93.0 / 93.0 |
| Qwen3-8B INT8 (community GPTQ) | 2 / 2 / failed to load | 91.5 / 93.0 / failed to load |
| Qwen3-8B INT4-AWQ | 4 / 6 / 4 † | 91.0 / 92.0 / 91.5 † |

† TensorRT-LLM could not load Qwen's published AWQ checkpoint, so that value is for an INT4-AWQ checkpoint quantized
here with TensorRT Model Optimizer (`scripts/modelopt_quantize.py`). It is a different checkpoint from the other two
columns. Qwen3-8B BF16 shows 0 because at about 32 ms per token it cannot meet an 8 s step; the failed INT8 cell
records both attempts in the report. Prefix caching matters for the 8B model too: with it off, the INT4-AWQ capacity
at the 8 s SLO falls from 4, 6 and 4 to 2, 1 and 2.

All the tables (every engine and model variant, caching on and off, tool-calling accuracy, run-to-run noise and
the test matrix with the cells that could not run) are in [reports/results.md](reports/results.md). The equal
scheduler settings comparison is in `reports/aligned/`. The written analysis is
[reports/REPORT.md](reports/REPORT.md).

![Latency against concurrency, Qwen3-4B BF16](reports/latency_qwen3-4b-bf16.png)

*Latency against concurrency, Qwen3-4B BF16. Solid lines are the agent workload, dashed lines chat; colours are
engines. The red line is the SLO and the dotted yellow line the relaxed step limit. Capacity against accuracy for
every engine and variant is in [reports/pareto_accuracy_capacity.png](reports/pareto_accuracy_capacity.png).*

## Quickstart

```bash
uv sync
scripts/collect_env.sh > results/env.txt       # record the hardware and software
uv run apg workload --profile agent            # seeded sessions into workloads/ (also: chat, fixed)
uv run python scripts/prepare_bfcl.py          # the 200-case tool-calling subset
servers/vllm.sh qwen3-4b-bf16                  # start one engine (one at a time)
uv run apg sweep --engine vllm --variant qwen3-4b-bf16 --workload agent \
  --engine-version vllm==0.32.0                # or --image <image> when the engine runs in a container
uv run apg quality --engine vllm --variant qwen3-4b-bf16
uv run apg report                              # tables and plots in reports/
uv run apg plan --users 200 --slo interactive  # slices needed, and the cost
```

Other commands: `apg run` (one concurrency level), `apg cell` (record why an engine and variant
can't run), `apg crosscheck` and `scripts/crosscheck_aiperf.sh` (compare the client with AIPerf).
The planner's worked example is in [docs/planner.md](docs/planner.md).

## Layout

```
configs/      model variants, engine settings, SLOs, workloads, test matrix, quality eval
servers/      one launch script per engine (container or venv)
src/agents_per_gpu/   workload generator, load client, metrics, sweeps, quality, report, planner
data/quality/ the fixed BFCL subset used for tool-calling accuracy
workloads/    generated, seeded session files (not committed; `apg workload` rebuilds them)
results/      raw per-request measurements (JSONL), the environment snapshot, quality runs, and the failed-cell
              notes; `results/aligned/` holds the equal-scheduler-settings runs and `results/a40-16q/` a few early
              runs on a different (time-sliced 16 GB) setup, kept apart and not used in the tables
reports/      plots and the written report
docs/         progress log and environment notes
```

## Limitations

- These results come from one whole A40, one GPU generation (Ampere). They don't transfer
  to H100 or Blackwell numbers, where FP8 and more memory change the picture.
- The workload is synthetic. It is shaped like agent traffic, but it is not a production
  trace.
- The engines evolve quickly. Each result records the engine version it was measured with. Engines ran from venvs,
  so there are no container image digests.
- Engines ran with their default settings, apart from one equal-settings check; none was tuned for this workload.

## License

MIT
