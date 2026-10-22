"""`apg run` and `apg sweep`: drive one engine and save raw JSONL, one file per run."""

import asyncio
import datetime as dt
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from agents_per_gpu import __version__, config
from agents_per_gpu.client import RequestSpec, RunAborted, run_level
from agents_per_gpu.metrics import slo_ratio, summarize
from agents_per_gpu.workload import load_workload, workload_path


@dataclass
class Target:
    engine: str
    variant: str
    workload: str
    prefix_cache: bool
    base_url: str = "http://localhost:8000"
    image: str | None = None
    engine_version: str | None = None
    quant_path: str | None = None
    results_dir: Path = Path("results")
    workloads_dir: Path = Path("workloads")
    config_dir: Path = config.DEFAULT_CONFIG_DIR


def run_path(results_dir: Path, t: Target, concurrency: int, repeat: int) -> Path:
    cache = "on" if t.prefix_cache else "off"
    name = f"{t.workload}-pc{cache}-c{concurrency:03d}-r{repeat}.jsonl"
    return Path(results_dir) / "raw" / t.engine / t.variant / name


def write_run(path: Path, run: dict[str, Any]) -> None:
    """Write atomically; raw results are append-only, so an existing file is never replaced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps({"type": "meta", **run["meta"]}) + "\n")
        for rec in run["requests"] + run["sessions"]:
            f.write(json.dumps(rec) + "\n")
    os.link(tmp, path)  # fails if path exists
    tmp.unlink()


def image_digest(image: str | None) -> str | None:
    if not image:
        return None
    try:
        out = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{index .RepoDigests 0}}", image],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def served_model(base_url: str) -> str:
    r = httpx.get(f"{base_url}/v1/models", timeout=10, trust_env=False)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


class Context:
    """Everything a run needs, resolved once per `run`/`sweep` invocation."""

    def __init__(self, t: Target):
        self.t = t
        self.matrix = config.load("matrix", t.config_dir)
        models = config.load("models", t.config_dir)
        workloads = config.load("workloads", t.config_dir)
        self.slo = config.load("slo", t.config_dir)

        if t.variant not in models["variants"]:
            raise SystemExit(f"unknown variant {t.variant!r}; see configs/models.yaml")
        variant = models["variants"][t.variant]

        path = workload_path(t.workloads_dir, t.workload, workloads["seed"])
        if not path.exists():
            raise SystemExit(f"{path} not found; run: apg workload --profile {t.workload}")
        self.workload_meta, self.items = load_workload(path)

        model = served_model(t.base_url)
        req = self.matrix["request"]
        extra = {**req["extra_body"], **(req.get("engine_extra_body") or {}).get(t.engine, {})}
        self.spec = RequestSpec(model, req["temperature"], extra)
        self.base_meta: dict[str, Any] = {
            "engine": t.engine,
            "variant": t.variant,
            "model": model,
            "hf_id": variant["hf_id"],
            "dtype": variant["dtype"],
            "quant_path": t.quant_path or (variant.get("quant_path") or {}).get(t.engine),
            "prefix_cache": t.prefix_cache,
            "image": t.image,
            "image_digest": image_digest(t.image),
            "engine_version": t.engine_version,
            "serving": self.matrix["serving"],
            "request": {"temperature": req["temperature"], "extra_body": extra},
            "slo_profiles": self.slo["profiles"],
            "max_error_rate": self.slo["max_error_rate"],
            "client_version": __version__,
            "date": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        }
        self.timeout_s = req["timeout_s"]
        self.warmup_timeout_s = req["warmup_timeout_s"]


async def _run_one(ctx: Context, concurrency: int, repeat: int, measure_seconds: float) -> dict:
    limits = httpx.Limits(max_connections=concurrency + 8, max_keepalive_connections=concurrency)
    timeout = httpx.Timeout(ctx.timeout_s, connect=10)
    async with httpx.AsyncClient(
        base_url=ctx.t.base_url, timeout=timeout, limits=limits, trust_env=False
    ) as http:
        run = await run_level(
            http,
            ctx.spec,
            ctx.workload_meta,
            ctx.items,
            concurrency=concurrency,
            repeat=repeat,
            warmup_requests=ctx.matrix["warmup_requests"],
            warmup_timeout_s=ctx.warmup_timeout_s,
            measure_seconds=measure_seconds,
            start_stagger_s=ctx.matrix.get("start_stagger_s", 0.0),
        )
    run["meta"] = {**ctx.base_meta, **run["meta"]}
    return run


def execute_run(
    ctx: Context, concurrency: int, repeat: int, measure_seconds: float | None = None
) -> dict[str, Any] | None:
    """Run one level and save it. Returns its summary, or None if the file already exists."""
    path = run_path(ctx.t.results_dir, ctx.t, concurrency, repeat)
    if path.exists():
        print(f"skip (exists): {path}", file=sys.stderr)
        return None
    seconds = measure_seconds if measure_seconds is not None else ctx.matrix["measure_seconds"]
    run = asyncio.run(_run_one(ctx, concurrency, repeat, seconds))
    write_run(path, run)
    s = summarize(run)
    print(
        f"{ctx.t.engine} {ctx.t.variant} {ctx.t.workload} c={concurrency} r={repeat}: "
        f"step p95 {s['step_p95_s']:.2f}s, ttft p95 {s['ttft_p95_s']:.2f}s, "
        f"itl p95 {s['itl_p95_ms']:.0f}ms, {s['output_tok_s']:.0f} tok/s, "
        f"errors {s['error_rate']:.1%} -> {path}",
        file=sys.stderr,
    )
    return s


def should_stop(level_summaries: list[dict], profiles: dict, factor: float) -> bool:
    """Stop the ladder once every SLO profile is broken by more than `factor` in some repeat."""
    return all(max(slo_ratio(s, p) for s in level_summaries) > factor for p in profiles.values())


def sweep(
    t: Target,
    concurrencies: list[int] | None = None,
    repeats: int | None = None,
    measure_seconds: float | None = None,
) -> int:
    ctx = Context(t)
    ladder = concurrencies or ctx.matrix["concurrency"]
    n_repeats = repeats or ctx.matrix["repeats"]
    factor = ctx.matrix["stop_when_slo_exceeded_by"]
    ran = 0
    for conc in ladder:
        level = []
        try:
            for rep in range(1, n_repeats + 1):
                s = execute_run(ctx, conc, rep, measure_seconds)
                if s is not None:
                    level.append(s)
                    ran += 1
        except RunAborted as e:
            print(f"aborted at c={conc}: {e}", file=sys.stderr)
            return 1
        if level and should_stop(level, ctx.slo["profiles"], factor):
            print(f"stopping: SLO exceeded by more than {factor}x at c={conc}", file=sys.stderr)
            break
    print(f"{ran} run(s) written under {Path(t.results_dir) / 'raw'}", file=sys.stderr)
    return 0
