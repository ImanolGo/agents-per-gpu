# How many agents can one A40 serve? vLLM, SGLang and TensorRT-LLM on agent traffic

Measured 2026-10-21 and 2026-10-22 on a whole NVIDIA A40 (48 GB, Ampere, no FP8), rented by the hour.
Every number below comes from `reports/results.md`, which `apg report` regenerates from the raw run files
in `results/`. `reports/aligned/results.md` holds the equal-settings runs.

## The question

Serving benchmarks usually report peak tokens per second on chat traffic. An agent behaves differently:
every request repeats a long system prompt and tool schema (about 2,000 tokens), the context grows with each
tool result, outputs are short, and a step is only sent after the previous one finishes. So I measured the
number that matters to an operator: **how many concurrent agent sessions fit on one GPU before a latency SLO
breaks, and what a task then costs.** The SLOs are my choices, not standards:

- `interactive`: p95 time to first token (TTFT) at most 1 s and p95 inter-token latency (ITL) at most 50 ms.
- `agent-step-relaxed`: p95 step latency (request sent to last token) at most 8 s. A stricter 3 s step SLO
  is also in the config, but nothing meets it: a 400-token final answer alone takes about 7 s at 17 to 19 ms
  per token, so it is reported as 0 everywhere and ignored below.

Capacity is the highest concurrency, in active sessions, before the first SLO failure on the ladder, taken as
the worst of the repeats.

## Setup

- **Engines** (venv installs, not containers, so no image digests): vLLM 0.32.0, SGLang 0.5.9,
  TensorRT-LLM 1.2.1 (PyTorch backend, `trtllm-serve`). Each used 0.90 of GPU memory and a 16,384-token context.
- **Models:** Qwen3-4B in BF16, and Qwen3-8B in BF16, GPTQ weight-only INT8 and AWQ INT4. Non-thinking mode,
  greedy decoding, fixed output lengths (`ignore_eos`).
- **Workloads:** `agent` (seeded sessions of 3 to 8 steps, a shared 2,000-token prefix, growing context) and
  `chat` (single-turn, no shared prefix, same total tokens), as the control.
- **Client:** `apg`, an async streaming closed-loop client. I checked it against NVIDIA AIPerf 0.13.0 at
  concurrency 1, 8 and 32 on all three engines (one run per level): output tokens per second agreed within 5 %
  in eight of nine cases, with one TensorRT-LLM run at 5.1 %. TTFT agreed within 5 % at concurrency 1 for vLLM
  (0.3 %) and SGLang (2.3 %) but not TensorRT-LLM (6.1 %), and differed more at 8 and 32 because the two clients
  space their requests differently.
- **Protocol:** 60 s windows after warm-up, 3 repeats per level. For the 4B agent sweeps with caching on I took
  3 more repeats on a second day, so those capacities are the worst of 6.
- **Price:** 0.535 USD per hour, a rental-market offer seen on 2026-10-21, not a list price (`TODO: verify`).

## Results

### Prefix caching decides agent capacity

Qwen3-4B BF16, agent workload, worst of the repeats (capacity at `interactive`; cost at that load):

| engine | caching off | caching on | USD per task, off / on |
|---|---|---|---|
| vLLM | 2 | 6 | 0.0018 / 0.0006 |
| SGLang | 3 | 6 | 0.0013 / 0.0005 |
| TensorRT-LLM | 2 | 6 | 0.0018 / 0.0006 |

Turning caching on triples capacity on vLLM and TensorRT-LLM and doubles it on SGLang, and cuts the cost
per task to between a third and two fifths. On chat traffic caching changes nothing (capacity 8, 12 and 12 either
way), as it should when prompts share no prefix. At concurrency 8, in the first repeat, vLLM's TTFT p95 falls
from 1.31 s to 0.35 s with caching and its throughput rises from 129 to 237 tokens per second.

The effect also holds for the quantized 8B model (INT4-AWQ, agent workload, the only 8B variant I ran with caching
off). At the 8 s step SLO capacity is 2, 1 and 2 without caching (vLLM, SGLang, TensorRT-LLM†) and 4, 6 and 4 with
it. At the `interactive` SLO it is 0 for all three without caching, because TTFT p95 at a single session is
already 1.11, 1.38 and 1.16 s (limit 1 s), against 0.30, 0.37 and 0.32 s with caching: every step re-prefills the
shared prefix and the whole context.

### The engines are close; SGLang leads at the 8 s step SLO

| 4B BF16, agent, caching on | `interactive` | step p95 at most 8 s | output tok/s at the 8 s capacity |
|---|---|---|---|
| vLLM | 6 | 3 | 137 |
| SGLang | 6 | 4 | 171 |
| TensorRT-LLM | 6 | 2 | 90 |

On chat, SGLang and TensorRT-LLM hold the `interactive` SLO to 12 sessions and vLLM to 8. The ranking held
across a second day (capacity identical at every rung except one SGLang repeat that reached 8 against 6 at
`interactive`; in single repeats, step p95 at c=8 for vLLM was 10.67 s on day 1 and 10.65 s on day 2) and when I gave all three engines the same explicit scheduler limits (8,192 tokens per step,
128 sequences, chunked prefill on): capacity and throughput at capacity were unchanged. Run-to-run variation
over six runs was at most 1.9 % in throughput and 1.5 % in step p95. I did not tune any engine beyond that
(CUDA graphs, attention backends and speculative decoding are untouched), so this is a result about defaults.

### A quantized 8B beats a BF16 4B

Qwen3-8B in BF16 records capacity 0 on agent traffic at both SLOs. At about 32 ms per token it cannot meet the
8 s step SLO at all, and at `interactive` the 0 comes from failing the ITL limit at c=1 (61 ms on vLLM) although
c=2 would pass, so that 0 understates it. The AWQ INT4 8B does serve agents:

| agent, caching on | vLLM | SGLang | TensorRT-LLM † |
|---|---|---|---|
| 4B BF16, step p95 at most 8 s | 3 | 4 | 2 |
| 8B INT4-AWQ, step p95 at most 8 s | 4 | 6 | 4 |
| 8B INT4-AWQ, `interactive` | 4 | 6 | 6 |
| 8B INT8 (GPTQ), step p95 at most 8 s | 2 | 2 | failed |

† TensorRT-LLM could not load Qwen's published AWQ checkpoint, so that column is an INT4-AWQ checkpoint I
quantized myself with TensorRT Model Optimizer (128 calibration samples). It is a different checkpoint from the
other two columns and the report marks it.

Tool-calling accuracy on a fixed 200-case BFCL-style subset (a simplified checker, not leaderboard scores,
and one point is two cases): 4B BF16 91.0 on all three engines; 8B BF16 92.0 on vLLM and 93.0 on SGLang and
TensorRT-LLM; 8B INT4-AWQ 91.0, 92.0 and 91.5†. No variant is more than 3 points below the 8B BF16 reference.
So on this workload the quantized 8B gives more concurrent agents at the same accuracy. The INT8 result is for
a community GPTQ weight-only checkpoint and says nothing about INT8 kernels in general.

## Why the curves look like this

These are hypotheses that fit the data, not findings I isolated.

- Without caching every step re-prefills 2,000 tokens plus the growing context, and those prefills stall the
  decode of every other session. That would explain why ITL p95 on vLLM jumps from 20 ms at c=2 to 82 ms at c=4
  with caching off, and why throughput flattens near 130 tokens per second from c=6. With caching on the same
  ladder reaches 237 tokens per second at c=8.
- Decode on a 4B model is memory-bound at about 17 to 19 ms per token on this card, so per-session speed is
  nearly the same on all engines at low concurrency. The engines differ in how well they hide prefill behind decode.
- I did not find why TensorRT-LLM is lowest at the 8 s step SLO. Its single-stream decode looked slower in a
  short smoke run, but I did not test that.

## Things the benchmark itself got wrong first

- The AIPerf comparison was confounded by prefix caching: AIPerf's synthetic prompts repeat and hit the cache
  (61 % of prompt tokens at c=32 on vLLM), mine do not. The cross-check has to run with caching off.
- Closed-loop workers with fixed output lengths fall into lock-step, so their prefills collide. The client now
  staggers each worker's first request.
- TensorRT-LLM with default settings rejects any prompt over 8,192 tokens (HTTP 400) because chunked prefill is
  off, and agent prompts reach 11,192 tokens. I enabled chunked prefill, which the other two engines have by
  default, and all TensorRT-LLM data was taken after that.
- TensorRT-LLM 1.2.1 failed to load both published quantized checkpoints (an assertion in its weight loader:
  the checkpoints store `qweight` tensors and no plain `weight`). A checkpoint quantized with Model Optimizer to
  INT4-AWQ loads; INT8 SmoothQuant exports but is rejected (`unsupported quant mode`). The INT8 cell stays failed.

## Limitations

- One GPU model, one day pair, one host class, default engine settings. A tuned engine could rank differently.
- The workload is synthetic and the SLOs are my choices. Capacity is measured on a coarse ladder (1, 2, 3, 4,
  6, 8, 12, 16, 24 and higher), so a one-rung difference is a difference between neighbouring rungs.
- Only the Qwen3-4B model and the 8B INT4-AWQ agent workload were measured with prefix caching off; the other 8B
  variants ran with caching on only, to save rental time.
- The accuracy subset has 200 cases. Differences of one or two points are within noise.
- Engines ran from venvs, so there are no container digests, which the repo's own rules ask for.
- This GPU is a whole A40. An earlier plan used a 16 GB time-sliced slice; those few runs are kept apart in
  `results/a40-16q/` and not mixed in.

## What would change on other hardware

Hypotheses only, not tested. Hopper and Blackwell add FP8 and FP4, so the quantization comparison would change.
More memory bandwidth shortens decode and lets the 8B BF16 model into the picture, and a bigger KV cache
reduces the penalty of running without prefix caching.

## Reproduce

```bash
uv sync
uv run apg workload --profile agent && uv run apg workload --profile chat
ENGINE_VENV=$HOME/venvs/vllm PREFIX_CACHE=1 servers/vllm.sh qwen3-4b-bf16     # then, in another shell:
uv run apg sweep --engine vllm --variant qwen3-4b-bf16 --workload agent --prefix-cache on \
  --engine-version vllm==0.32.0
uv run apg report        # regenerates reports/results.md from results/
uv run apg plan --users 100 --slo agent-step-relaxed
```
