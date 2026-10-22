"""Cross-check the apg client against NVIDIA AIPerf on one fixed-length config (PLAN.md phase 1)."""

import json
from pathlib import Path
from typing import Any

from agents_per_gpu.metrics import load_run

TOLERANCE = 0.05


def _aiperf_stat(data: dict[str, Any], tag: str, stat: str = "avg") -> float | None:
    """AIPerf reports TTFT in ms; convert via the unit field. Layout: {tag: {unit, avg, ...}}."""
    metric = data.get(tag)
    if not isinstance(metric, dict) or stat not in metric:
        return None
    value = float(metric[stat])
    return value / 1000 if str(metric.get("unit", "")).lower() == "ms" else value


def client_numbers(run_path: Path) -> dict[str, float]:
    run = load_run(run_path)
    ok = [r for r in run["requests"] if r["phase"] == "measure" and r["success"]]
    window = run["meta"]["window_end"] - run["meta"]["window_start"]
    return {
        "ttft_mean_s": sum(r["ttft_s"] for r in ok) / len(ok),
        "output_tok_s": sum(r["output_tokens"] for r in ok) / window,
    }


def compare(client: dict[str, float], aiperf_json: Path) -> list[dict[str, Any]]:
    data = json.loads(Path(aiperf_json).read_text())
    pairs = [
        ("TTFT mean (s)", client["ttft_mean_s"], _aiperf_stat(data, "time_to_first_token")),
        ("output tok/s", client["output_tok_s"], _aiperf_stat(data, "output_token_throughput")),
    ]
    rows = []
    for name, ours, theirs in pairs:
        diff = None if theirs in (None, 0) else (ours - theirs) / theirs
        rows.append(
            {
                "metric": name,
                "apg": ours,
                "aiperf": theirs,
                "diff": diff,
                "within_tolerance": diff is not None and abs(diff) <= TOLERANCE,
            }
        )
    return rows


def render(rows: list[dict[str, Any]]) -> str:
    lines = ["| metric | apg | aiperf | diff | within 5% |", "|---|---|---|---|---|"]
    for r in rows:
        aiperf = "missing" if r["aiperf"] is None else f"{r['aiperf']:.4g}"
        diff = "n/a" if r["diff"] is None else f"{r['diff']:+.1%}"
        lines.append(
            f"| {r['metric']} | {r['apg']:.4g} | {aiperf} | {diff} | "
            f"{'yes' if r['within_tolerance'] else 'NO'} |"
        )
    return "\n".join(lines)
