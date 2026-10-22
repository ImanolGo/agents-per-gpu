"""Turn raw results into capacity rows, cost and quality lookups. Shared by report and plan."""

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from agents_per_gpu.metrics import capacity, iter_run_files, load_run, summarize
from agents_per_gpu.quality import load_quality

GroupKey = tuple[str, str, str, bool]  # engine, variant, workload, prefix_cache


def load_summaries(results_dir: Path) -> list[dict[str, Any]]:
    return [summarize(load_run(p)) for p in iter_run_files(results_dir)]


def group_summaries(summaries: list[dict[str, Any]]) -> dict[GroupKey, list[dict[str, Any]]]:
    groups: dict[GroupKey, list[dict[str, Any]]] = defaultdict(list)
    for s in summaries:
        groups[(s["engine"], s["variant"], s["workload"], s["prefix_cache"])].append(s)
    return dict(sorted(groups.items()))


def capacity_rows(
    summaries: list[dict[str, Any]], slo_cfg: dict[str, Any], profile: str
) -> list[dict[str, Any]]:
    rows = []
    for (engine, variant, workload, prefix_cache), runs in group_summaries(summaries).items():
        cap = capacity(runs, slo_cfg["profiles"][profile], slo_cfg["max_error_rate"])
        rows.append(
            {
                "engine": engine,
                "variant": variant,
                "workload": workload,
                "prefix_cache": prefix_cache,
                "profile": profile,
                "levels": sorted({r["concurrency"] for r in runs}),
                **cap,
            }
        )
    return rows


def hourly_slice_usd(
    matrix_cfg: dict[str, Any], a40_usd_per_hour: float | None = None
) -> float | None:
    """Cost of one slice per hour (a whole GPU at slice_fraction 1.0), or None without a price."""
    price = (
        a40_usd_per_hour if a40_usd_per_hour is not None else matrix_cfg["cost"]["a40_usd_per_hour"]
    )
    return None if price is None else price * matrix_cfg["cost"]["slice_fraction"]


def add_cost(row: dict[str, Any], hourly: float | None) -> dict[str, Any]:
    """$/1M output tokens and $/agent task at capacity (from the worst repeat's throughput)."""
    out = dict(row, usd_per_m_tokens=None, usd_per_task=None)
    if hourly is None or not row.get("capacity"):
        return out
    if row.get("output_tok_s"):
        out["usd_per_m_tokens"] = hourly / (row["output_tok_s"] * 3600) * 1e6
    if row.get("sessions_per_hour"):
        out["usd_per_task"] = hourly / row["sessions_per_hour"]
    return out


def latest_quality(results_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Newest quality run per (engine, variant); file names sort by timestamp."""
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted((Path(results_dir) / "quality").glob("*/*.jsonl")):
        meta, _ = load_quality(path)
        latest[(meta["engine"], meta["variant"])] = meta
    return latest


def read_cells(results_dir: Path) -> list[dict[str, Any]]:
    path = Path(results_dir) / "cells.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_cell(results_dir: Path, record: dict[str, Any]) -> None:
    path = Path(results_dir) / "cells.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def fmt(value: float | None, spec: str = ".1f", missing: str = "n/a") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return missing
    return format(value, spec)
