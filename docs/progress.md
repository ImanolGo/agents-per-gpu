# Progress log

Newest entry first. One entry per work session: what was done, what was measured, what is still open.

## 2026-10-22: README graphics

- Added `src/agents_per_gpu/figures.py`: a prefix-caching headline (agents per GPU off and on, cost per task) and
  a model-size comparison (agents at the 8 s step SLO with tool-calling accuracy). `apg report` now writes them
  as `reports/hero_prefix_caching.png` and `reports/model_size.png`; no number changed in `reports/results.md`.
- Rewrote the top of the README in plain words, with the findings and both graphics.
- Tests: one new test for the figures; ruff and pytest pass (71 tests).

## 2026-10-22 (release): history squashed, plan and README brought up to date

- The public history is one commit. The earlier 23 local commits were squashed before the first push because
  three of them named the rental provider, which `AGENTS.md` forbids in the history. The tree was identical
  before and after (same tree hash), so no content changed. The day-by-day record is in this log.
- `PLAN.md` status line, release checklist and the `TODO: verify` markers were brought up to date. One marker is
  deliberately left: the GPU price (0.535 USD/hour) is a rental-market offer, not a verified list price.
- README results now include the 8B variants, the accuracy table and the caching-off 8B result.
- **One hand edit to result files, disclosed:** the `dir` field inside each AIPerf `profile_export_aiperf.json` held the
  absolute path on the rental machine; it was replaced with `~/...` before committing, and nothing else in those
  files was changed. Raw run files (`results/raw`, `results/aligned/raw`, `results/quality`) are exactly as written.
  AIPerf's own logs, inputs and console output are not committed because they hold local paths.
- **Correction found during the release check:** the early A40-16Q session had in fact committed AIPerf's full output
  (logs, phase exports, server metrics, inputs) under `results/a40-16q/aiperf/`, and those files contained a username
  and home path from the old workstation. They were in the first public commit. This commit removes everything there
  except each level's summary `profile_export_aiperf.json` and `.csv`, which is all `apg crosscheck` reads.


## 2026-10-21 (night): moved to a rented whole A40

The A40-16Q VM is no longer used. Benchmarks now run on a whole NVIDIA A40 (48 GB, Ampere GA102,
driver 595.99, CUDA 13.0 toolkit) rented by the hour, in a container, so engines run from venvs.

Changes to the plan (PLAN.md, README and configs updated to match):
- No 16 GB cap. `gpu_memory_fraction` stays 0.90 of 48 GB, so the KV cache is about 3x larger than before.
- `qwen3-8b-bf16` is now a variant in the matrix and the quality reference is measured, not borrowed.
- `cost.slice_fraction` is 1.0: a "slice" in `apg plan` and `apg report` is the whole GPU.
  `a40_usd_per_hour` is a rental-market price (0.535, seen 2026-10-21); `TODO: verify` before publishing.
- The earlier vLLM numbers (Qwen3-4B BF16, A40-16Q vGPU) are a different setup. Keep them labelled as
  such; do not mix them with whole-A40 runs in one table. The `agent-step-relaxed` SLO was chosen
  after that run, so re-check whether the plain `agent-step` SLO is reachable on the whole GPU.
- Tests (69) and ruff pass after the change.

### Phase 1 on the whole A40: vLLM

- vLLM 0.32.0 (torch 2.13, CUDA 13.0 runtime, driver 595.99) serves Qwen3-4B BF16 from a venv. The
  KV-cache and memory settings follow `configs/matrix.yaml` (0.90 of 48 GB). `results/env.txt` is the
  new snapshot (AMD EPYC 7662 host; the old one is in `results/a40-16q/`, with the old raw runs).
- `ignore_eos` is honoured: 32 of 32 requests had `length_ok` at concurrency 1.
- Single-stream decode is about 17-19 ms/token (57-58 tok/s), close to the 16Q slice's 19 ms.
- **AIPerf cross-check, vLLM, prefix caching off, fixed 1,000-in / 128-out, 60 s each, one run:**

  | c | TTFT apg / AIPerf (s) | diff | output tok/s apg / AIPerf | diff |
  |---|---|---|---|---|
  | 1 | 0.0987 / 0.0985 | +0.3 % | 57.6 / 57.45 | +0.3 % |
  | 8 | 0.139 / 0.344 | -59.6 % | 332.8 / 340.5 | -2.3 % |
  | 32 | 0.564 / 0.601 | -6.1 % | 682.7 / 678.5 | +0.6 % |

  Throughput agrees within 5 % at all three levels; TTFT agrees at c=1 only (c=32 is 6.1 %, c=8 is a
  different arrival pattern). The acceptance criterion says "or the difference is explained".

**What the first cross-check attempts showed (all of it matters for reading later results):**
1. **Prefix caching confounds the comparison.** With the server's prefix caching on, AIPerf's synthetic
   prompts repeat and hit the cache (vLLM counters: 61 % of prompt tokens at c=32, 4 % at c=8) while
   apg's per-run nonce gives 0 %. Result: apg looked 12-35 % slower at c=8 and c=32. The vLLM server-side
   histograms (mean ITL 31 vs 42 ms, prefill 187 vs 392 ms) pinned this down. The cross-check
   must run with caching off (`PREFIX_CACHE=0`, `APG_PREFIX_CACHE=off`); `scripts/crosscheck_aiperf.sh`
   now takes that option and says why.
2. **Closed-loop workers with fixed-length outputs fall into lock-step.** Without a stagger, all 8
   workers sent within 0.4 s of each other every 3 s, so their prefills collided and TTFT p95 was
   inflated (apg 0.36 s vs AIPerf p50 0.08 s). `run_level` now spreads the first request of worker
   `w` by `w/C x start_stagger_s` (2 s in `matrix.yaml`, recorded in each run's meta). The agent
   workload's variable lengths desynchronise sessions on their own; the fixed profile does not.
   Runs made before this change (including everything in `results/a40-16q/`) have no stagger.
3. **Run-to-run noise at c=8 is large on this host**: apg's output tok/s over repeated identical runs
   was 324, 333, 361 and 416 (the latter two with caching on; not all runs were under the same server
   state). Do not read one run as a result: the 3-repeat rule and the noise table (phase 2) matter.

The client is not CPU-bound (about 20 % of one core at c=32, 38 % at c=8).

### Phase 2 on the whole A40: vLLM, Qwen3-4B BF16, agent and chat, prefix caching off and on

Ladders 1-24 (agent stops at 8 with caching off and 16 with it on; chat at 24 for both), 3 repeats of
60 s per level, one day, 98 run files in `results/raw/` (the 3 `fixed` ones are the AIPerf cross-check).
No request errors in any run. All numbers regenerate with `apg report`.

| Capacity (worst of 3) | agent, cache off | agent, cache on | chat, cache off | chat, cache on |
|---|---|---|---|---|
| `interactive` (TTFT p95 <= 1 s, ITL p95 <= 50 ms) | 2 | **6** | 8 | 8 |
| `agent-step-relaxed` (step p95 <= 8 s) | 2 | 3 | 3 | 3 |
| `agent-step` (step p95 <= 3 s) | 0 | 0 | 0 | 0 |

- **Prefix caching matters for agent traffic and not for chat**, as expected: at c=8 TTFT p95 is 0.35 s
  with it against 1.31 s without, and output tok/s is 237 against 129. Chat with a unique prompt per
  request is the control and does not move (same 72 requests and 16,157 tokens in the c=8 window;
  step p95 10.71 vs 10.72 s).
- Cost at the (unverified) rental price of 0.535 USD/h, agent traffic at `interactive` capacity:
  0.73 USD per 1M output tokens and 0.0006 USD per task with caching, 1.91 and 0.0018 without.
- `agent-step` (3 s) is unreachable for every row: a 400-token final answer alone takes about 7 s at
  17-19 ms/token. It was already unreachable on the A40-16Q. Decide whether to drop it or replace it
  with a per-token budget; it currently reports 0 everywhere.
- Repeat-to-repeat variation was small (CV of step p95 under 1.5 % at every level, under 0.5 % in
  throughput except c=12 at 1.3 %). One day only: the plan's multi-day noise check is still open.

Known issues in this data:
- `agent`, caching on, c=1 has **2 repeats, not 3**: repeat 1 was skipped because a 30 s smoke-test file
  of the same name existed on the VM (not committed). Re-run c=1 with `--repeats 3` after deleting it there.
- The `fixed` calibration runs appear in `apg report` as a workload, including a misleading `32+`
  capacity. Exclude them from the report before publishing.

### SGLang 0.5.9, Qwen3-4B BF16 (same protocol and settings as vLLM; whole A40)

SGLang launched first time from its venv (radix cache off via `--disable-radix-cache`, on by default).
`ignore_eos` honoured (32 of 32 `length_ok`). AIPerf cross-check, caching off: output tok/s within 1 %
at c=1, 8, 32 (+0.1, -0.9, +0.6 %); TTFT +2.3 % at c=1, -25 % at c=8, -10 % at c=32 (client arrival
pattern, as for vLLM). Agent and chat sweeps, caching off and on, 3 repeats, no errors.

| Capacity (worst of 3) | agent off | agent on | chat off | chat on |
|---|---|---|---|---|
| SGLang, `interactive` | 3 | 6 | 12 | 12 |
| vLLM, `interactive` | 2 | 6 | 8 | 8 |
| SGLang, `agent-step-relaxed` | 2 | 4 | 3 | 3 |
| vLLM, `agent-step-relaxed` | 2 | 3 | 3 | 3 |

- On agent traffic with caching on, SGLang serves 4 sessions at the 8 s step SLO against vLLM's 3, and
  both serve 6 at `interactive`. At c=8 (first repeats): 260 vs 237 tok/s, step p95 9.9 vs 10.9 s.
  Capacity is a coarse ladder (levels 4, 6, 8), so the 4-vs-3 gap sits between neighbouring rungs.
- On chat, SGLang holds `interactive` to c=12 against 8 for vLLM. Chat output is unchanged by caching,
  as expected for prompts with no shared prefix.
- Both engines ran with 0.90 of GPU memory and a 16,384-token context but different internal defaults
  (SGLang: chunked prefill 4,096 and CUDA graphs up to batch 32; vLLM: its own scheduler defaults, not
  tuned). Per AGENTS.md, no tuning was done for either, so differences may partly be defaults.
  `TODO` before publishing: tune both equally, or say so in the report.

### TensorRT-LLM 1.2.1 (PyTorch backend, `trtllm-serve` from a venv), Qwen3-4B BF16

Launched from the venv without trouble. KV cache 230,112 tokens (SGLang 233,884), 0.90 free-memory
fraction, block reuse off/on with `PREFIX_CACHE`.

- **Real finding, config difference:** with its default settings `trtllm-serve` has chunked prefill off,
  and `max_num_tokens` is 8,192. A prompt above that is rejected with HTTP 400 ("prompt length (8793)
  should not exceed max_num_tokens (8192)"). Agent prompts grow past 8,192 tokens in late steps (up to
  11,192 here), so the first smoke run lost 2 of 30 requests. `servers/trtllm.sh` now sets
  `enable_chunked_prefill: true`, which vLLM and SGLang use by default. After the change a c=4 agent run
  had 0 failures in 94 requests, 5 of them above 8,192 tokens. All sweep data was taken after this change.
  The cross-check (1,000-token prompts, unaffected) ran before it.
- `ignore_eos` honoured. AIPerf cross-check, caching off: output tok/s -2.1 %, -5.1 %, -4.7 % at c=1, 8,
  32 (c=8 is just outside the 5 % bar); TTFT +6.1 % at c=1 (also just outside), -69 % and -51 % at c=8, 32
  (client arrival pattern). Not repeated, so the c=1 and c=8 misses may be noise.
- Agent and chat sweeps, caching off and on, 3 repeats, no request errors in any run.

**Three engines, Qwen3-4B BF16, whole A40, worst of 3 repeats:**

| Capacity | agent off | agent on | chat off | chat on |
|---|---|---|---|---|
| `interactive`: vLLM | 2 | 6 | 8 | 8 |
| `interactive`: SGLang | 3 | 6 | 12 | 12 |
| `interactive`: TensorRT-LLM | 2 | 6 | 12 | 12 |
| `agent-step-relaxed`: vLLM | 2 | 3 | 3 | 3 |
| `agent-step-relaxed`: SGLang | 2 | 4 | 3 | 3 |
| `agent-step-relaxed`: TensorRT-LLM | 1 | 2 | 3 | 3 |

- At `interactive` the three engines are close on agent traffic with caching (6 each); on chat SGLang
  and TensorRT-LLM hold 12 against vLLM's 8.
- At the 8 s step SLO TensorRT-LLM is the lowest on agent traffic (1 off, 2 on). **Not explained.**
  Candidates, none tested: its single-stream decode looked slower in a 30 s smoke run (36 tok/s, ITL
  p95 21 ms, against 44-55 tok/s and 18 ms elsewhere), it was run with untuned defaults (PyTorch backend,
  `max_num_tokens` 8,192), and capacity is a coarse ladder. Do not publish a ranking from this
  without tuning all three engines equally and repeating on a second day.
- Everything above is one day, one GPU, default engine settings.

### 8B variants, quality and capacity on all three engines (2026-10-21/22, new A40 host, same specs)

The first A40 instance lost its GPU while stopped (the provider re-let it: "could take hours to weeks" to free up), so I
destroyed it and rented another identical-spec A40 (same GPU, driver 595.99, CPU and OS lines in `env.txt`
identical). Venvs rebuilt with the same pins; one transitive package differed (`zipp` 4.1.2 vs 4.1.1).
Everything below used **prefix caching on only** for the 8B variants, to save credit. 3 repeats, 60 s,
no request errors in any run. `qwen3-4b-bf16` got a quality run only (its sweeps already existed).

**Tool-calling accuracy** (200 BFCL-style cases, simplified checker, so 1 point = 2 cases; reference =
vLLM Qwen3-8B BF16 = 92.0 %, set in `configs/quality.yaml`):

| variant | vLLM | SGLang | TensorRT-LLM |
|---|---|---|---|
| qwen3-4b-bf16 | 91.0 | 91.0 | 91.0 |
| qwen3-8b-bf16 | 92.0 | 93.0 | 93.0 |
| qwen3-8b-int8 (GPTQ) | 91.5 | 93.0 | cell failed |
| qwen3-8b-int4-awq | 91.0 | 92.0 | cell failed |

No variant is flagged (none is more than 3 points below the reference). Differences of 1-2 points are
within what 200 cases resolve; do not read a ranking from them. SGLang and TensorRT-LLM scoring 1 point
above vLLM on the same BF16 weights is the size of the noise, or of tiny numeric differences between engines.

**Capacity, caching on (worst of 3; tok/s at that level)**

| interactive SLO | vLLM agent | SGLang agent | TRT-LLM agent | vLLM chat | SGLang chat | TRT-LLM chat |
|---|---|---|---|---|---|---|
| 4B BF16 | 6 (203) | 6 (220) | 6 (193) | 8 | 12 | 12 |
| 8B BF16 | 0 | 0 | 0 | 6 | 6 | 8 |
| 8B INT8 | 4 (136) | 4 (135) | failed | 3 | 0 | failed |
| 8B INT4-AWQ | 4 (200) | 6 (243) | failed | 6 | 4 | failed |

| 8 s step SLO | vLLM agent | SGLang agent | TRT-LLM agent |
|---|---|---|---|
| 4B BF16 | 3 | 4 | 2 |
| 8B BF16 | 0 | 0 | 0 |
| 8B INT8 | 2 | 2 | failed |
| 8B INT4-AWQ | 4 | 6 | failed |

- **8B BF16 shows 0 for two different reasons.** At the 8 s step SLO it is physically unreachable: about
  32 ms/token, so a 400-token final answer alone is ~13 s. At the `interactive` agent SLO it fails the ITL
  limit at c=1 (p95 61 ms on vLLM) although c=2 would pass (43 ms); capacity stops at the first failing
  level, so 0 understates it. The capacity rule and the SLO choice both disadvantage the largest model.
- **INT4-AWQ 8B beats 4B BF16 on agent capacity at the 8 s SLO** on vLLM (4 vs 3) and SGLang (6 vs 4), with
  accuracy equal or higher (91-92 vs 91). That is the answer to README question 2 for these two engines,
  on this workload and GPU.
- **INT8 (GPTQ weight-only) is slower than INT4-AWQ** on vLLM and SGLang (agent 4 vs 4-6 at `interactive`).
  The checkpoint is a community GPTQ one, not W8A8, so this says nothing about INT8 kernels in general.
- **TensorRT-LLM cannot load either quantized checkpoint.** Both die in the weight loader with
  `AssertionError` at `assert "weight" in weights[0]` (linear.py `load_weights_vanilla_helper`, layer
  o_proj). Verified from the checkpoints' safetensors index: they hold `qweight`, `qzeros`, `scales` (and
  `g_idx` for GPTQ) and no plain `weight`, so TensorRT-LLM 1.2.1 took the unquantized path. The cell
  records this. Not tried: quantizing with TensorRT Model Optimizer (the plan's alternative path for
  TRT-LLM), or loading the checkpoints with a different quantization setting. Another engine's checkpoint
  failing in TRT-LLM's own PyTorch loader is a result about the stock community checkpoints, not about
  TensorRT-LLM's quantization support.
- The first reason string `apg cell` recorded was the shutdown error `AttributeError: ... no attribute
  'cuda_graph_runner'` (a follow-on crash), not the cause. I replaced it by hand in `results/cells.jsonl`
  after reading the log. The driver's error grep should skip `Exception ignored in` lines.

### Second day and equal scheduler settings, Qwen3-4B BF16, agent workload, prefix caching on (2026-10-22)

**Second day, original settings.** Repeats 4-6 of the agent sweep for each engine, taken about 17 to 20
hours after repeats 1-3 on the same instance type (a new host of the same spec for the 8B work; the 4B
day-1 runs are from the first host). So the "worst of n" is now n=6 over 2 days.
- Capacity is unchanged in every case: `interactive` 6 for all three engines (SGLang had one repeat that
  reached 8; the worst of 6 is 6), 8 s step SLO: vLLM 3, SGLang 4, TensorRT-LLM 2.
- Run-to-run noise over the 2 days (6 runs per level): CV of output tok/s up to 1.9 % and of step p95 up
  to 1.5 % (TensorRT-LLM, c=4 and c=8); vLLM and SGLang are lower, mostly under 0.5 %.
  The 10 % trigger in PLAN.md phase 2 is far away, so 3 repeats per level were enough.
- Cross-check of single runs: step p95 at c=8 is 10.67 s (day 1) vs 10.65 s (day 2) for vLLM, 9.78 vs 9.76 s
  for SGLang. The ranking by step p95 at c=8 (SGLang 9.8, vLLM 10.7, TensorRT-LLM 12.3 s) is the same on both days.
- A race in my driver made the first TensorRT-LLM second-day launch fail ("default launch FAILED trtllm"):
  `servers/trtllm.sh` is a wrapper that later becomes `trtllm-serve`, and a one-second liveness check
  saw no `trtllm-serve` process yet. The server was fine. The phase was re-run with a 30 s grace period.

**Equal scheduler settings** (`ALIGNED=1` in the launch scripts; results in `results/aligned/`, report in
`reports/aligned/`). Every engine got the same explicit limits: 8,192 tokens per scheduler step, at most 128
concurrent sequences, chunked prefill on, prefix caching on. Verified in the server logs: vLLM
`max_num_batched_tokens=8192`, `max_num_seqs=128`; SGLang `chunked_prefill_size=8192`,
`max_running_requests=128`; TensorRT-LLM `max_num_tokens=8192`, `max_batch_size=128`, chunked prefill on.
The defaults they replace were: vLLM unset (resolved at run time), SGLang chunk 4,096 and 4,096 running
requests, TensorRT-LLM batch size 2,048 and 8,192 tokens.
- Capacity and throughput at capacity are identical to the default-settings run for all three engines
  (`interactive` 6, 8 s step SLO 3 / 4 / 2; the same tok/s to the digit except SGLang 171 vs 173).
  Step p95 at c=8 and c=16 differ by under 0.3 s between the two settings.
- Reading: at concurrency up to 24 on a 4B model these scheduler limits are not binding, so the engine
  ranking on this workload does not come from them. This is not tuning in the sense of searching for
  each engine's best configuration (CUDA graphs, attention backend, speculative decoding and so on were
  not touched), so it does not show that a tuned engine could not rank differently.

### TensorRT-LLM with checkpoints quantized here (TensorRT Model Optimizer 0.37.0), 2026-10-22

The two published quantized checkpoints do not load in TensorRT-LLM 1.2.1 (see above), so I quantized
Qwen3-8B myself with Model Optimizer (PTQ with 128 `cnn_dailymail`
samples of up to 512 tokens, then `export_hf_checkpoint`), now `scripts/modelopt_quantize.py`.
- **INT4-AWQ (`INT4_AWQ_CFG`, W4A16, group 128): works.** Quantized in 12 minutes, 5.8 GB, loads and serves.
  Accuracy 91.5 % on the 200 cases (vLLM and SGLang with Qwen's AWQ checkpoint: 91.0 and 92.0 %).
  Agent capacity, caching on: 6 at `interactive`, 4 at the 8 s step SLO (vLLM 4 and 4, SGLang 6 and 6 with the
  published checkpoint). Chat: 8 at `interactive`. These runs carry the quantization in `quant_path`, and the
  report marks them with a dagger and a footnote, because they are a different checkpoint from the other
  two engines'. The run files' `hf_id` field still says `Qwen/Qwen3-8B-AWQ` (it comes from `models.yaml`);
  `quant_path` and `model` (`int4awq`) are the true description.
- **INT8 SmoothQuant (`INT8_SMOOTHQUANT_CFG`, W8A8): exports in 68 s, is rejected by the server**:
  `ValueError: unsupported quant mode: [14]` in `linear.py get_quant_method` (the PyTorch backend has no linear
  method for `W8A8_SQ_PER_CHANNEL`). Model Optimizer 0.37.0 has no INT8 weight-only config, the closest match to
  the community GPTQ INT8, so I stopped there. The INT8 cell stays failed, with both attempts in its reason.
- Honesty about this comparison: an AWQ checkpoint I made with 128 calibration samples is not Qwen's, so the
  TensorRT-LLM AWQ numbers say that TensorRT-LLM can serve INT4-AWQ, not how it compares like-for-like.

### Written report, README and plot fixes (2026-10-22)

`reports/REPORT.md` is written (about 1,600 words). Every figure in it was checked against
`reports/results.md`; four statements were wrong or too strong in the first draft and were corrected
(AIPerf TTFT agreement at c=1 holds for vLLM and SGLang but not TensorRT-LLM, 6.1 %; the cost reduction from
caching is "between a third and two fifths", not "a third or less"; the 8B BF16 zero at `interactive` is a rule
artefact; the ladder is open-ended above 24). The README banner now carries the headline. The accuracy-against-
capacity plot had overlapping labels, so it now uses colour for the engine, shape for the variant and a
legend. A fresh clone plus `apg report` regenerates `reports/results.md`, `runs.csv` and the equal-settings
report exactly (diff clean apart from the generation date), and the 70 tests pass.

### 8B INT4-AWQ, agent workload, prefix caching off (2026-10-22)

The last measurement, on all three engines (TensorRT-LLM with the own-quantization checkpoint, marked in the
report). 3 repeats, no errors. Capacity (worst of 3), caching off then on:
- `agent-step-relaxed`: vLLM 2 then 4, SGLang 1 then 6 (repeats 1, 1, 2), TensorRT-LLM 2 then 4.
- `interactive`: 0 on all three without caching, 4 / 6 / 6 with it. The zero is a plain TTFT failure at c=1:
  p95 1.11, 1.38 and 1.16 s against a 1 s limit; with caching 0.30, 0.37 and 0.32 s.
The only 8B workload and variant measured with caching off; the other 8B variants and chat were not.
Credit used for the whole session after the restart: about 5 USD. The rental instance was stopped and deleted
once every result was copied back and the file listings matched (783 raw runs, 11 quality files, 78 equal-
settings files).

Open (not done, and not needed for the headline): 8B BF16, INT8 and chat with caching off; a TensorRT-LLM INT8 path
(Model Optimizer has no weight-only INT8 config in 0.37.0); real tuning of each engine (CUDA graphs, attention
backend, speculative decoding), which could change the ranking; a longer noise study on a third day; the `agent-step`
3 s profile, which nothing meets; container image digests (engines ran from venvs); and pushing the repo, which
should follow a history squash because three local commits name the rental provider.
Everything is stopped: no server, sweep or benchmark process is running and the GPU is idle.

### Setup used (deviations from PLAN.md)

- **Engines run from venvs, not containers.** `/var` (Docker's root) had 17 GB free and the engine
  images are larger, so nothing was pulled. Venvs live in `~/venvs/{vllm,sglang,trtllm,aiperf}`
  (not in the repo). Versions are in `results/env.txt`: vLLM 0.32.0 (torch 2.13, CUDA 13.0),
  SGLang 0.5.9, TensorRT-LLM 1.2.1, AIPerf 0.13.0. Launch with
  `ENGINE_VENV=$HOME/venvs/<engine> servers/<engine>.sh <variant>`. Results record
  `engine_version` (pass `--engine-version`) instead of an image digest. This breaks the
  "pinned container digest" requirement in AGENTS.md; it is a known gap, not a choice.
- **vLLM needs `VLLM_USE_FLASHINFER_SAMPLER=0`** (set in `servers/vllm.sh` for venv runs). The
  FlashInfer sampler failed at JIT build; its log says nvcc or ninja is missing. Greedy decoding
  (temperature 0) makes the sampler choice irrelevant here, but it is a difference to note.
- **TensorRT-LLM venv:** `pip install tensorrt_llm` needed `pyarrow<19` (it pins NumPy 1.x).
  `import tensorrt_llm` works. The server has never been launched.
- **SGLang venv:** installed, never launched or imported successfully in a clean check.
- **Measurement window 60 s** (PLAN.md said 180) and 3 repeats, to fit the matrix. The ladder
  gained levels 3, 6 and 12 in `configs/matrix.yaml`; only the first sweep (1, 2, 4, 8, 16) ran
  with the old ladder, so those three levels are not measured yet.
- **INT8 variant uses `JunHowie/Qwen3-8B-GPTQ-Int8`** (community, weight-only GPTQ INT8, Apache-2.0).
  Qwen publishes no INT8 Qwen3-8B. This is not the W8A8 PLAN.md described. TRT-LLM may not load it.
- Both 8B checkpoints (`Qwen/Qwen3-8B-AWQ`, the GPTQ-Int8 one) are already in the Hugging Face cache.
- **Post-hoc SLO:** `agent-step` (3 s) is unreachable on this vGPU, so `agent-step-relaxed`
  (8 s) was added after seeing data and is the `headline_profile`. Both are reported.

### Measured (vLLM 0.32.0, Qwen3-4B BF16, prefix caching on, max_model_len 16384, GPU memory 0.90)

All numbers regenerate with `uv run apg report` (`reports/results.md`, `reports/runs.csv`).

- **AIPerf cross-check passed on vLLM** at concurrency 1, 8 and 32 with the fixed 1,000-in /
  128-out workload: TTFT within 3.8%, 0.2% and 4.1%; output tok/s within 0.9%, 1.3% and 0.7%.
  `ignore_eos` is honoured (`length_ok` true on every checked request). The AIPerf JSON layout
  `{tag: {unit, avg}}` assumed by `crosscheck.py` is correct.
- **Single-stream decode is about 19 ms/token (about 50 tok/s)** on this A40-16Q slice.
- **KV cache is small:** vLLM reported 5.24 GiB, 38,128 tokens, for Qwen3-4B BF16 at 0.90.
- **Agent workload, 3 repeats per level** (step p95 / ITL p95 / output tok/s):
  c=1 5.9 s / 19 ms / 50; c=2 6.3 s / 21 ms / 97; c=4 8.3 s / 27 ms / 157;
  c=8 11.4 s / 56 ms / 230; c=16 21.7 s / 216 ms / 185 (TTFT p95 9.5 s). The ladder stopped at 16
  under the 2x rule. Throughput falls from 8 to 16; the cause (KV pressure) is a hypothesis.
- **Chat workload, c=1, 2, 4, 8 only** (I stopped it before 16): step p95 6.3, 7.3, 8.8, 10.8 s;
  ITL p95 18, 19, 21, 41 ms; 49, 98, 172, 269 tok/s.
- **Capacity (worst of 3):** agent at `interactive` = 4 (the next level, 8, has ITL 56 ms; 6 not
  run); agent at `agent-step-relaxed` = 2; agent at `agent-step` = 0 (a 400-token answer alone
  takes about 7.6 s). The chat row `8+` in the `interactive` table is an artifact of the stopped
  sweep, not a real capacity.
- **Run-to-run spread was small** (step p95 of the repeats mostly within 1-3%, up to about 5% at
  c=8) during this window, which is
  encouraging but covers one evening only. The multi-day noise study (PLAN.md phase 2 item 4) is
  not done.
- First-step prompts are about 2,100 tokens (2,000 system + task) and grow to 8,000+ by the last
  step, so context length is a large part of the load at this KV size.

### Not done

- vLLM: prefix caching off (restart with `PREFIX_CACHE=0`), ladder levels 3, 6, 12, the chat
  ladder from 16 up, `apg quality`, the 8B variants.
- SGLang and TensorRT-LLM: nothing run. Record any variant that can't load with `apg cell`.
- Noise table across days, a public GPU price, the Qwen3-8B BF16 reference accuracy, and the
  phase 5 write-up (`reports/REPORT.md`, README headline) are all open.
- Phase 1 acceptance is met only for vLLM: SGLang and TensorRT-LLM have not answered a streamed
  request, and `results/env.txt` was captured before any container pin existed.

### Where to continue

1. Start vLLM again (`ENGINE_VENV=$HOME/venvs/vllm servers/vllm.sh qwen3-4b-bf16`) and finish its
   4B runs: `apg sweep --engine vllm --variant qwen3-4b-bf16 --workload agent|chat
   --engine-version vllm==0.32.0` (existing run files are skipped, so this fills levels 3, 6, 12
   and the chat levels above 8), then `apg quality --engine vllm --variant qwen3-4b-bf16`.
2. Restart with `PREFIX_CACHE=0` and run the agent and chat sweeps with `--prefix-cache off`.
3. Do SGLang the same way (check the launch flags and `ignore_eos` first with `apg run` at
   concurrency 1: look at `length_ok`), then TensorRT-LLM. Repeat the AIPerf cross-check for each
   engine with `scripts/crosscheck_aiperf.sh` (`AIPERF=$HOME/venvs/aiperf/bin/aiperf`).
4. Serve each 8B variant on each engine that can load it. For a variant that can't, run
   `apg cell --engine E --variant V --reason "<verbatim error>"`.
5. Re-run the sweeps on a second and third day to get the noise table, then `apg report`,
   fill `cost.a40_usd_per_hour` (with source and date) and the quality reference, and write phase 5.
6. Server and sweep logs were written to `/tmp/apg-logs`, which is volatile; nothing in the repo
   depends on them.

Raw results of this session are committed under `results/raw/`. Of the AIPerf output only each
run's top-level `profile_export_aiperf.json` is committed; its output-directory path had the
local home directory replaced with `~` (the only edit to a result file), and the logs, inputs and
server-metrics files stay out of git because they hold local paths.

## 2026-10-21: tooling for phases 1-4

Done (code and tests only; **nothing has been measured on a real engine yet**):

- `apg workload`: seeded agent, chat and fixed-length workloads, sized with the Qwen3 tokenizer.
  Checked against the real tokenizer: 1, 50, 1,500 and 4,000-token texts and the 2,000-token system
  prompt (12 tool schemas) all come out at the exact requested count.
- `apg run` and `apg sweep`: async streaming closed-loop client. One JSONL file per run (meta line,
  then request and session lines), written atomically and never replaced. Per-request TTFT, ITL,
  step latency, token counts and a `length_ok` flag.
- `apg report`, `apg plan`, `apg cell`, `apg quality`, `apg crosscheck`, and
  `scripts/crosscheck_aiperf.sh`, `scripts/prepare_bfcl.py`.
- 69 tests pass, including client and sweep runs against an in-process fake engine. `ruff check` and
  `ruff format --check` pass.

Decisions that differ from or extend PLAN.md (PLAN.md is updated to match):

- Session context uses the workload's synthetic tool-call text, not the model's reply, so every
  engine sees identical prompts. Tool results are wrapped in `<tool_response>` tags.
- Warm-up ends after `max(warmup_requests, concurrency)` completions, so the window opens after the
  start-up burst. Warm-up and drain requests stay in the raw file, tagged, and are excluded from
  summaries.
- Each run puts a unique nonce after the shared system prompt (agent) or at the start (chat), so
  repeated runs can't reuse the prefix cache on session-specific text.
- Capacity per repeat is the last passing level before the first failing one. The reported value is
  the minimum across repeats. Throughput and cost use the minimum among repeats at that level.
- Quality uses a simplified BFCL-style checker on 100 `simple_python` and 100 `multiple` cases
  (seed 0). The model's tool-call parser must be on in the server, so the launch scripts enable it.

Verified from current docs (2026-10-21):

- `trtllm-serve`: `--max_seq_len`, `--config` (alias `--extra_llm_api_options`),
  `--free_gpu_memory_fraction` and `--tool_parser qwen3` exist.
- AIPerf: `aiperf profile` with `--streaming --endpoint-type chat --isl --osl --concurrency
  --benchmark-duration --warmup-request-count --extra-inputs --use-server-token-count --random-seed
  --output-artifact-dir`. Its README says output length is only guaranteed with `ignore_eos` passed
  through `--extra-inputs`.
- BFCL data layout and Apache-2.0 licence; Qwen3-4B, 8B and 8B-AWQ are Apache-2.0 on the Hub.

Not verified, and still open:

- The JSON layout of AIPerf's `profile_export_aiperf.json`. `crosscheck.py` assumes
  `{tag: {unit, avg}}` and reports a missing metric as a failure rather than guessing.
- Whether each engine honours `ignore_eos` on its chat endpoint. The client records `length_ok` per
  request, so the first real run shows it.
- vLLM `--tool-call-parser hermes` and SGLang `--tool-call-parser qwen25` for Qwen3, container image
  tags and digests, and the TRT-LLM `kv_cache_config` keys.
- A public A40 price, with source and date. Costs print `n/a` until `cost.a40_usd_per_hour` is set.
- The reference accuracy for Qwen3-8B BF16 (`configs/quality.yaml`).
- No engine image is pulled on the VM and `results/env.txt` is not committed. Run
  `scripts/collect_env.sh > results/env.txt` after pinning the images.

Next: pin image tags, start vLLM with Qwen3-4B BF16 and run `apg run` at concurrency 1, then the
AIPerf cross-check. Phase 1 acceptance is not met until that passes on all three engines.

## 2026-10-20: scaffold

- Repo created: README, PLAN, AGENTS, configs, engine launch scripts, `collect_env.sh` and the `apg` CLI skeleton.
- Prior work on the VM: TensorRT-LLM hello world with TinyLlama-1.1B-Chat. It isn't part of this repo.
- Open: everything in phase 1. Start by pinning the container images and running `collect_env.sh`.
