"""Build synthetic run dicts and result trees for tests."""

from pathlib import Path

from agents_per_gpu.sweep import Target, run_path, write_run


def make_run(
    *,
    concurrency=1,
    repeat=1,
    step_latencies=(1.0,),
    ttfts=None,
    itl=0.02,
    output_tokens=100,
    failures=0,
    window=10.0,
    sessions=0,
    task_latency=5.0,
    engine="vllm",
    variant="qwen3-4b-bf16",
    workload="agent",
    prefix_cache=True,
    date="2026-01-01T00:00:00+00:00",
):
    ttfts = ttfts or [0.2] * len(step_latencies)
    requests = []
    for lat, ttft in zip(step_latencies, ttfts, strict=True):
        requests.append(
            {
                "type": "request",
                "phase": "measure",
                "success": True,
                "ttft_s": ttft,
                "itl_s": [itl] * 5,
                "step_latency_s": lat,
                "output_tokens": output_tokens,
                "length_ok": True,
            }
        )
    for _ in range(failures):
        requests.append({"type": "request", "phase": "measure", "success": False})
    requests.append({"type": "request", "phase": "warmup", "success": True})
    return {
        "meta": {
            "engine": engine,
            "variant": variant,
            "workload": workload,
            "prefix_cache": prefix_cache,
            "concurrency": concurrency,
            "repeat": repeat,
            "date": date,
            "image_digest": f"{engine}@sha256:test",
            "window_start": 100.0,
            "window_end": 100.0 + window,
        },
        "requests": requests,
        "sessions": [
            {
                "type": "session",
                "phase": "measure",
                "complete": True,
                "task_latency_s": task_latency,
            }
            for _ in range(sessions)
        ],
    }


def write_ladder(
    results_dir: Path,
    ladder: dict[int, float],
    *,
    engine="vllm",
    variant="qwen3-4b-bf16",
    workload="agent",
    repeats=2,
    sessions=5,
    prefix_cache=True,
    dates=None,
    output_tokens=100,
):
    """One run file per (concurrency, repeat); `ladder` maps concurrency to step latency."""
    target = Target(engine, variant, workload, prefix_cache)
    for conc, latency in ladder.items():
        for rep in range(1, repeats + 1):
            run = make_run(
                concurrency=conc,
                repeat=rep,
                step_latencies=[latency] * 20,
                ttfts=[latency / 5] * 20,
                sessions=sessions,
                engine=engine,
                variant=variant,
                workload=workload,
                prefix_cache=prefix_cache,
                output_tokens=output_tokens,
                **({"date": dates[rep - 1]} if dates else {}),
            )
            write_run(run_path(results_dir, target, conc, rep), run)
