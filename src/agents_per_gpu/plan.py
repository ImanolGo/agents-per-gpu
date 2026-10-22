"""`apg plan`: size a deployment from measured capacity."""

import math
from typing import Any

from agents_per_gpu.analysis import fmt


def build_plan(
    rows: list[dict[str, Any]],
    users: int,
    sessions_per_user: float,
    hourly_slice_usd: float | None,
    accuracy: dict[tuple[str, str], float] | None = None,
    min_accuracy: float | None = None,
) -> dict[str, Any]:
    """Options sorted best first: fewest slices, then lowest hourly cost, then highest capacity.

    `rows` are capacity rows (with cost fields) for one workload, one SLO and one prefix-cache
    setting. Demand is `users * sessions_per_user` concurrent agent sessions.
    """
    demand = math.ceil(users * sessions_per_user)
    options, excluded = [], []
    for r in rows:
        acc = (accuracy or {}).get((r["engine"], r["variant"]))
        if not r.get("capacity"):
            excluded.append((r["engine"], r["variant"], "no capacity at this SLO"))
            continue
        if min_accuracy is not None and (acc is None or acc < min_accuracy):
            reason = (
                "no quality result" if acc is None else f"accuracy {acc:.1f}% < {min_accuracy}%"
            )
            excluded.append((r["engine"], r["variant"], reason))
            continue
        slices = math.ceil(demand / r["capacity"])
        options.append(
            {
                "engine": r["engine"],
                "variant": r["variant"],
                "capacity": r["capacity"],
                "censored": r.get("censored", False),
                "repeats": r["repeats"],
                "slices": slices,
                "utilization": demand / (slices * r["capacity"]),
                "usd_per_hour": None if hourly_slice_usd is None else slices * hourly_slice_usd,
                "usd_per_m_tokens": r.get("usd_per_m_tokens"),
                "usd_per_task": r.get("usd_per_task"),
                "accuracy": acc,
            }
        )
    options.sort(key=lambda o: (o["slices"], o["usd_per_hour"] or 0.0, -o["capacity"]))
    return {"demand": demand, "options": options, "excluded": excluded}


def render(plan: dict[str, Any], users: int, sessions_per_user: float, context: str) -> str:
    lines = [
        f"Demand: {users} users x {sessions_per_user} concurrent session(s) = "
        f"{plan['demand']} concurrent agent sessions ({context})",
        "",
        "| engine | variant | capacity | slices | utilisation | $/hour "
        "| $/1M out tok | $/task | acc % |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for o in plan["options"]:
        cap = f"{o['capacity']}{'+' if o['censored'] else ''} (n={o['repeats']})"
        lines.append(
            f"| {o['engine']} | {o['variant']} | {cap} | {o['slices']} | {o['utilization']:.0%} "
            f"| {fmt(o['usd_per_hour'], '.2f')} | {fmt(o['usd_per_m_tokens'], '.3f')} "
            f"| {fmt(o['usd_per_task'], '.4f')} | {fmt(o['accuracy'], '.1f')} |"
        )
    if not plan["options"]:
        lines.append("| (none) | | | | | | | | |")
    for engine, variant, reason in plan["excluded"]:
        lines.append(f"\nExcluded: {engine} / {variant}: {reason}")
    lines += [
        "",
        "Assumptions:",
        "- A slice is the unit priced by matrix.yaml `cost.slice_fraction`; "
        "capacity is the worst of the repeated runs.",
        "- Slices needed = ceil(concurrent sessions / capacity); load is spread evenly.",
        "- $/1M tokens and $/task are at capacity, from the lowest throughput among repeats; "
        "'n/a' while no GPU price is set.",
        "- A '+' on capacity means the ladder never broke the SLO: the true capacity is higher.",
    ]
    return "\n".join(lines)
