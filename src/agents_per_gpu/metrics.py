"""Raw run files, per-run summaries, SLO checks and capacity.

A run file is JSONL: one `meta` line, then `request` and `session` lines. Only records with
phase == "measure" count; warm-up and drain records stay in the file for traceability.
"""

import json
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

SLO_KEYS = ("ttft_p95_s", "itl_p95_ms", "step_p95_s")


def load_run(path: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    requests: list[dict] = []
    sessions: list[dict] = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            kind = rec.pop("type")
            if kind == "meta":
                meta = rec
            elif kind == "request":
                requests.append(rec)
            elif kind == "session":
                sessions.append(rec)
    return {"meta": meta, "requests": requests, "sessions": sessions}


def iter_run_files(results_dir: Path) -> Iterable[Path]:
    yield from sorted((Path(results_dir) / "raw").glob("*/*/*.jsonl"))


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else math.nan


def cv(values: list[float]) -> float:
    """Coefficient of variation (sample std / mean); NaN with fewer than two values."""
    if len(values) < 2 or np.mean(values) == 0:
        return math.nan
    return float(np.std(values, ddof=1) / np.mean(values))


def summarize(run: dict[str, Any]) -> dict[str, Any]:
    meta = run["meta"]
    window = meta["window_end"] - meta["window_start"]
    measured = [r for r in run["requests"] if r["phase"] == "measure"]
    ok = [r for r in measured if r["success"]]
    sessions = [s for s in run["sessions"] if s["phase"] == "measure" and s["complete"]]

    ttft = [r["ttft_s"] for r in ok]
    itl = [g for r in ok for g in r["itl_s"]]
    step = [r["step_latency_s"] for r in ok]
    task = [s["task_latency_s"] for s in sessions]

    out: dict[str, Any] = {
        "engine": meta["engine"],
        "variant": meta["variant"],
        "workload": meta["workload"],
        "prefix_cache": meta["prefix_cache"],
        "concurrency": meta["concurrency"],
        "repeat": meta["repeat"],
        "date": meta["date"],
        "image_digest": meta.get("image_digest"),
        "quant_path": meta.get("quant_path"),
        "workload_seed": meta.get("workload_seed"),
        "n_requests": len(measured),
        "n_failed": len(measured) - len(ok),
        "error_rate": (len(measured) - len(ok)) / len(measured) if measured else 1.0,
        "length_mismatch_rate": (
            sum(1 for r in ok if not r["length_ok"]) / len(ok) if ok else math.nan
        ),
        "output_tok_s": sum(r["output_tokens"] for r in ok) / window if window > 0 else math.nan,
        "requests_per_s": len(ok) / window if window > 0 else math.nan,
        "sessions_completed": len(sessions),
        "sessions_per_hour": len(sessions) / window * 3600 if window > 0 else math.nan,
    }
    for name, values, scale in (("ttft", ttft, 1.0), ("itl", itl, 1000.0), ("step", step, 1.0)):
        unit = "ms" if scale == 1000.0 else "s"
        for q in (50, 95, 99):
            out[f"{name}_p{q}_{unit}"] = percentile(values, q) * scale
    out["task_p50_s"] = percentile(task, 50)
    out["task_p95_s"] = percentile(task, 95)
    return out


def slo_ratio(summary: dict[str, Any], profile: dict[str, float]) -> float:
    """Worst observed/limit across the profile's limits; above 1 means the SLO is broken."""
    ratios = []
    for key, limit in profile.items():
        value = summary.get(key, math.nan)
        ratios.append(math.inf if math.isnan(value) else value / limit)
    return max(ratios)


def meets_slo(summary: dict[str, Any], profile: dict[str, float], max_error_rate: float) -> bool:
    if summary["n_requests"] == 0 or summary["error_rate"] >= max_error_rate:
        return False
    return slo_ratio(summary, profile) <= 1.0


def capacity(
    summaries: list[dict[str, Any]], profile: dict[str, float], max_error_rate: float
) -> dict[str, Any]:
    """Highest concurrency before the first SLO failure, as the worst across repeats.

    Within one repeat the ladder is walked upward and stops at the first failing level, so a
    lucky pass above a failure doesn't count. `censored` means the limiting repeat never failed
    (the true capacity is at least the reported value).
    """
    by_repeat: dict[int, list[dict]] = defaultdict(list)
    for s in summaries:
        by_repeat[s["repeat"]].append(s)

    per_repeat: dict[int, dict[str, Any]] = {}
    for rep, levels in by_repeat.items():
        cap, failed = 0, False
        for s in sorted(levels, key=lambda s: s["concurrency"]):
            if meets_slo(s, profile, max_error_rate):
                cap = s["concurrency"]
            else:
                failed = True
                break
        per_repeat[rep] = {"capacity": cap, "failed": failed}

    if not per_repeat:
        return {"capacity": None, "repeats": 0}

    worst = min(v["capacity"] for v in per_repeat.values())
    limiting = [r for r, v in per_repeat.items() if v["capacity"] == worst]
    at_cap = [s for s in summaries if s["concurrency"] == worst]
    result: dict[str, Any] = {
        "capacity": worst,
        "repeats": len(per_repeat),
        "per_repeat": {r: v["capacity"] for r, v in sorted(per_repeat.items())},
        "censored": any(not per_repeat[r]["failed"] for r in limiting),
        "output_tok_s": None,
        "sessions_per_hour": None,
        "task_p50_s": None,
    }
    if worst > 0 and at_cap:
        # Minimum across repeats keeps cost estimates conservative.
        result["output_tok_s"] = min(s["output_tok_s"] for s in at_cap)
        result["sessions_per_hour"] = min(s["sessions_per_hour"] for s in at_cap)
        result["task_p50_s"] = max(s["task_p50_s"] for s in at_cap)
    return result
