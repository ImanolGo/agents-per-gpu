"""Async streaming client and the closed-loop load generator.

All durations come from `time.perf_counter()`. Wall-clock time is only recorded as a label.
Each of C workers runs sessions back to back: a session's next step is sent only after the
previous one finishes, which is what makes the agent workload closed-loop.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from agents_per_gpu.workload import build_request, n_requests


class RunAborted(RuntimeError):
    pass


@dataclass
class RequestSpec:
    model: str
    temperature: float = 0.0
    extra_body: dict[str, Any] = field(default_factory=dict)


async def stream_chat(
    http: httpx.AsyncClient, spec: RequestSpec, messages: list[dict], max_tokens: int
) -> dict[str, Any]:
    payload = {
        "model": spec.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": spec.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        **spec.extra_body,
    }
    t_send = time.time()
    t0 = time.perf_counter()
    first = last = None
    gaps: list[float] = []
    chunks = 0
    usage: dict | None = None
    error: str | None = None
    try:
        async with http.stream("POST", "/v1/chat/completions", json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread())[:200].decode(errors="replace")
                error = f"http {resp.status_code}: {body}"
            else:
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    choices = chunk.get("choices") or []
                    delta = choices[0].get("delta") or {} if choices else {}
                    if delta.get("content") or delta.get("reasoning_content"):
                        now = time.perf_counter()
                        if first is None:
                            first = now
                        else:
                            gaps.append(now - last)
                        last = now
                        chunks += 1
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        error = f"{type(e).__name__}: {e}"[:200]

    t_end = time.perf_counter()
    if error is None and first is None:
        error = "no tokens received"
    ok = error is None
    out_tokens = (usage or {}).get("completion_tokens", chunks) if ok else 0
    return {
        "success": ok,
        "error": error,
        "t_send": t_send,
        "t_end_perf": t_end,
        "ttft_s": first - t0 if ok else None,
        "itl_s": [round(g, 6) for g in gaps] if ok else [],
        "step_latency_s": last - t0 if ok else t_end - t0,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "output_tokens": out_tokens,
        "output_tokens_source": "usage" if usage else "chunks",
        "expected_output_tokens": max_tokens,
        "length_ok": ok and out_tokens == max_tokens,
        "chunks": chunks,
    }


class _Run:
    def __init__(self, warmup_target: int, warmup_timeout_s: float, measure_seconds: float):
        self.t0 = time.perf_counter()
        self.wall0 = time.time()
        self.warmup_target = warmup_target
        self.warmup_timeout_s = warmup_timeout_s
        self.measure_seconds = measure_seconds
        self.warm_done = 0
        self.warm_ok = 0
        self.window_start: float | None = None
        self.window_end = 0.0
        self.stop = False
        self.requests: list[dict] = []
        self.sessions: list[dict] = []

    def wall(self, perf: float) -> float:
        return self.wall0 + (perf - self.t0)

    def classify(self, t_end: float, success: bool) -> str:
        if self.window_start is None:
            self.warm_done += 1
            self.warm_ok += success
            if self.warm_done >= self.warmup_target or t_end - self.t0 >= self.warmup_timeout_s:
                self.window_start = t_end
                self.window_end = t_end + self.measure_seconds
            return "warmup"
        if t_end <= self.window_end:
            return "measure"
        self.stop = True
        return "drain"


async def run_level(
    http: httpx.AsyncClient,
    spec: RequestSpec,
    meta: dict[str, Any],
    items: list[dict[str, Any]],
    *,
    concurrency: int,
    repeat: int,
    warmup_requests: int,
    warmup_timeout_s: float,
    measure_seconds: float,
    start_stagger_s: float = 0.0,
) -> dict[str, Any]:
    """Run one concurrency level. Returns {"meta", "requests", "sessions"} (meta: window times)."""
    # At least one full round, so the window opens after the start-up burst has settled.
    run = _Run(max(warmup_requests, concurrency), warmup_timeout_s, measure_seconds)

    async def session(worker: int, idx: int) -> None:
        item = items[idx % len(items)]
        nonce = f"{repeat}.{concurrency}.{idx}"
        started = time.perf_counter()
        complete, phase, steps = True, "warmup", 0
        for step in range(n_requests(meta, item)):
            if run.stop:
                complete = False
                break
            messages, max_tokens = build_request(meta, item, step, nonce)
            res = await stream_chat(http, spec, messages, max_tokens)
            phase = run.classify(res.pop("t_end_perf"), res["success"])
            run.requests.append(
                {"type": "request", "phase": phase, "worker": worker, "session": item["id"]}
                | {"step": step, "t_end": time.time()}
                | res
            )
            steps += 1
            if not res["success"]:
                complete = False
                await asyncio.sleep(0.2)  # don't spin against a dead server
                break
        if steps == 0:
            return
        run.sessions.append(
            {
                "type": "session",
                "phase": phase,
                "worker": worker,
                "session": item["id"],
                "steps": steps,
                "complete": complete,
                "task_latency_s": time.perf_counter() - started,
                "t_end": time.time(),
            }
        )

    async def worker(w: int) -> None:
        # Spread the first requests so fixed-length closed-loop workers don't stay in lock-step,
        # which makes every prefill collide (see docs/progress.md, 2026-10-21 night).
        await asyncio.sleep(start_stagger_s * w / concurrency)
        j = 0
        while not run.stop:
            await session(w, w + concurrency * j)
            j += 1
            if run.window_start is not None and run.warm_ok == 0:
                run.stop = True
                return

    await asyncio.gather(*(worker(w) for w in range(concurrency)))
    if run.window_start is None or run.warm_ok == 0:
        raise RunAborted("no request succeeded during warm-up; is the server up?")
    return {
        "meta": {
            **meta_for_run(meta),
            "concurrency": concurrency,
            "repeat": repeat,
            "window_start": run.wall(run.window_start),
            "window_end": run.wall(run.window_end),
            "measure_seconds": measure_seconds,
            "warmup_target": run.warmup_target,
            "start_stagger_s": start_stagger_s,
        },
        "requests": run.requests,
        "sessions": run.sessions,
    }


def meta_for_run(workload_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "workload": workload_meta["kind"],
        "workload_seed": workload_meta["seed"],
        "tokenizer": workload_meta["tokenizer"],
    }
