"""`apg report`: rebuild every table and plot in reports/ from results/ and configs/."""

import datetime as dt
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from agents_per_gpu import config  # noqa: E402
from agents_per_gpu.analysis import (  # noqa: E402
    add_cost,
    capacity_rows,
    fmt,
    group_summaries,
    hourly_slice_usd,
    latest_quality,
    load_summaries,
    read_cells,
)
from agents_per_gpu.figures import plot_caching_hero, plot_model_size  # noqa: E402
from agents_per_gpu.metrics import cv  # noqa: E402

NOISE_FLAG = 0.10  # PLAN.md: above this, raise the number of repeats


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No data._\n"
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def _cap_cell(row: dict[str, Any]) -> str:
    cap = row["capacity"]
    if cap is None:
        return "n/a"
    per = ", ".join(str(v) for v in row["per_repeat"].values())
    return f"{cap}{'+' if row['censored'] else ''} (n={row['repeats']}: {per})"


def capacity_section(summaries, slo_cfg, hourly) -> str:
    out = []
    own = _own_quant(summaries)
    for profile, limits in slo_cfg["profiles"].items():
        limit_text = ", ".join(f"{k} <= {v}" for k, v in limits.items())
        rows = [add_cost(r, hourly) for r in capacity_rows(summaries, slo_cfg, profile)]
        table = md_table(
            [
                "engine",
                "variant",
                "workload",
                "prefix cache",
                "capacity (worst of n)",
                "output tok/s",
                "sessions/h",
                "$/1M out tok",
                "$/task",
            ],
            [
                [
                    r["engine"],
                    r["variant"] + (" \u2020" if (r["engine"], r["variant"]) in own else ""),
                    r["workload"],
                    "on" if r["prefix_cache"] else "off",
                    _cap_cell(r),
                    fmt(r["output_tok_s"], ".0f"),
                    fmt(r["sessions_per_hour"], ".0f"),
                    fmt(r["usd_per_m_tokens"], ".3f"),
                    fmt(r["usd_per_task"], ".4f"),
                ]
                for r in rows
            ],
        )
        out.append(
            f"### SLO profile `{profile}` ({limit_text}; error rate < "
            f"{slo_cfg['max_error_rate']:.0%})\n\n{table}"
        )
    out.append(
        "Capacity is the highest concurrency, in active sessions, before the first SLO failure "
        "on the ladder, taken as the worst of the repeated runs. A `+` means the ladder never "
        "broke the SLO for the limiting run, so capacity is at least that value. Throughput "
        "and cost use the lowest value among repeats at the capacity level.\n" + _own_footnote(own)
    )
    return "\n".join(out)


# A run whose quant_path says this used a checkpoint quantized here, not a published one.
OWN_PTQ = "own PTQ"


def _own_quant(items) -> dict[tuple[str, str], str]:
    """(engine, variant) -> quant_path for runs on a checkpoint we quantized ourselves."""
    return {
        (i["engine"], i["variant"]): i["quant_path"]
        for i in items
        if OWN_PTQ in (i.get("quant_path") or "")
    }


def _own_footnote(own: dict[tuple[str, str], str]) -> str:
    notes = sorted({qp for qp in own.values()})
    return "".join(f"\n\u2020 own quantization, not a published checkpoint: {qp}\n" for qp in notes)


def matrix_section(summaries, slo_cfg, matrix_cfg, cells) -> str:
    headline = slo_cfg.get("headline_profile", "agent-step")
    caps = {
        (r["engine"], r["variant"]): r
        for r in capacity_rows(
            [s for s in summaries if s["workload"] == "agent" and s["prefix_cache"]],
            slo_cfg,
            headline,
        )
    }
    latest_cell = {(c["engine"], c["variant"]): c for c in cells}
    own = _own_quant(summaries)
    rows = []
    for variant in matrix_cfg["variants"]:
        row = [variant]
        for engine in matrix_cfg["engines"]:
            if (engine, variant) in caps:
                row.append(
                    _cap_cell(caps[(engine, variant)])
                    + (" \u2020" if (engine, variant) in own else "")
                )
            elif (engine, variant) in latest_cell:
                c = latest_cell[(engine, variant)]
                row.append(f"{c['status']}: {c['reason']}")
            else:
                row.append("not run")
        rows.append(row)
    return md_table(["variant"] + list(matrix_cfg["engines"]), rows) + (
        f"\nCells show agent-workload capacity at the `{headline}` SLO with prefix caching on. "
        "A cell that could not run keeps its status and reason; it is a result.\n"
        + _own_footnote(own)
    )


def quality_section(results_dir, qcfg) -> str:
    latest = latest_quality(results_dir)
    ref = qcfg["reference"]
    ref_acc = ref["accuracy"]
    rows = []
    own = _own_quant([{"engine": e, "variant": v, **m} for (e, v), m in latest.items()])
    for (engine, variant), m in sorted(latest.items()):
        acc = m["accuracy"] * 100
        if ref_acc is None:
            delta, flag = "reference not set", ""
        else:
            delta = f"{acc - ref_acc:+.1f}"
            flag = "FLAG" if ref_acc - acc > qcfg["flag_drop_points"] else ""
        rows.append(
            [
                engine,
                variant + (" \u2020" if (engine, variant) in own else ""),
                f"{acc:.1f}",
                f"{m.get('simple', math.nan) * 100:.1f}",
                f"{m.get('multiple', math.nan) * 100:.1f}",
                m["n"],
                delta,
                flag,
                m["date"][:10],
            ]
        )
    note = (
        f"\nReference ({ref['variant']}): "
        + (
            "not set; it needs a published score or a hosted BF16 run on the same cases.\n"
            if ref_acc is None
            else f"{ref_acc:.1f}% from {ref['source']}. Flag = more than "
            f"{qcfg['flag_drop_points']} points below it.\n"
        )
        + "Scores use a simplified BFCL-style checker on a fixed subset; they are not "
        "leaderboard numbers.\n" + _own_footnote(own)
    )
    return (
        md_table(
            [
                "engine",
                "variant",
                "accuracy %",
                "simple %",
                "multiple %",
                "cases",
                "vs reference (points)",
                "flag",
                "date",
            ],
            rows,
        )
        + note
    )


def noise_section(summaries) -> str:
    by_cfg: dict[tuple, list[dict]] = defaultdict(list)
    for s in summaries:
        by_cfg[
            (s["engine"], s["variant"], s["workload"], s["prefix_cache"], s["concurrency"])
        ].append(s)
    rows = []
    for key, runs in sorted(by_cfg.items()):
        if len(runs) < 2:
            continue
        tok, step = (cv([r[m] for r in runs]) for m in ("output_tok_s", "step_p95_s"))
        flag = "raise repeats" if max(tok, step) > NOISE_FLAG else ""
        engine, variant, workload, pc, conc = key
        rows.append(
            [
                engine,
                variant,
                workload,
                "on" if pc else "off",
                conc,
                len(runs),
                len({r["date"][:10] for r in runs}),
                fmt(tok * 100, ".1f"),
                fmt(step * 100, ".1f"),
                flag,
            ]
        )
    return md_table(
        [
            "engine",
            "variant",
            "workload",
            "prefix cache",
            "concurrency",
            "runs",
            "days",
            "CV tok/s %",
            "CV step p95 %",
            "note",
        ],
        rows,
    ) + (
        f"\nCV = sample standard deviation / mean across repeated runs of the same config. "
        f"Above {NOISE_FLAG:.0%} PLAN.md says to raise the number of repeats.\n"
    )


def plot_latency(summaries, slo_cfg, variant, path: Path) -> Path | None:
    runs = [s for s in summaries if s["variant"] == variant and s["workload"] in ("agent", "chat")]
    if not runs:
        return None
    profiles = slo_cfg["profiles"]
    panels = [
        ("step_p95_s", "step latency p95 (s)", profiles["agent-step"]["step_p95_s"]),
        ("ttft_p95_s", "TTFT p95 (s)", profiles["interactive"]["ttft_p95_s"]),
        ("itl_p95_ms", "ITL p95 (ms)", profiles["interactive"]["itl_p95_ms"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = {e: f"C{i}" for i, e in enumerate(sorted({s["engine"] for s in runs}))}
    for (key, metric_label, limit), ax in zip(panels, axes, strict=True):
        for (engine, _, workload, pc), group in group_summaries(runs).items():
            by_c: dict[int, list[float]] = defaultdict(list)
            for s in group:
                by_c[s["concurrency"]].append(s[key])
            xs = sorted(by_c)
            ys = [float(pd.Series(by_c[c]).median()) for c in xs]
            lo = [min(by_c[c]) for c in xs]
            hi = [max(by_c[c]) for c in xs]
            style = "-" if workload == "agent" else "--"
            ax.plot(
                xs,
                ys,
                style,
                marker="o",
                color=colors[engine],
                label=f"{engine} {workload} pc-{'on' if pc else 'off'}",
            )
            ax.fill_between(xs, lo, hi, color=colors[engine], alpha=0.12)
        if limit:
            ax.axhline(limit, color="red", lw=1, ls=":", label="SLO limit")
        if key == "step_p95_s":
            relaxed = profiles["agent-step-relaxed"]["step_p95_s"]
            ax.axhline(relaxed, color="orange", lw=1, ls=":", label="relaxed limit (post-hoc)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("concurrent sessions")
        ax.set_ylabel(metric_label)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{variant}: median across repeats, band = min..max")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_pareto(summaries, slo_cfg, quality, path: Path) -> Path | None:
    headline = slo_cfg.get("headline_profile", "agent-step")
    caps = capacity_rows(
        [s for s in summaries if s["workload"] == "agent" and s["prefix_cache"]],
        slo_cfg,
        headline,
    )
    points = [
        (
            r["engine"],
            r["variant"],
            r["capacity"],
            quality[(r["engine"], r["variant"])]["accuracy"] * 100,
        )
        for r in caps
        if r["capacity"] is not None and (r["engine"], r["variant"]) in quality
    ]
    if not points:
        return None
    engines = sorted({e for e, *_ in points})
    variants = sorted({v for _, v, *_ in points})
    colors = {
        e: f"C{i}" for i, e in enumerate(engines)
    }  # same order and colours as the latency plots
    markers = dict(zip(variants, "osD^v<>p", strict=False))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for engine, variant, x, y in points:
        # a small horizontal offset per engine keeps points at the same (capacity, accuracy) visible
        dx = (engines.index(engine) - (len(engines) - 1) / 2) * 0.12
        ax.scatter(x + dx, y, color=colors[engine], marker=markers[variant], s=55, alpha=0.9)
    handles = [plt.Line2D([], [], color=colors[e], marker="o", ls="", label=e) for e in engines]
    handles += [
        plt.Line2D([], [], color="0.35", marker=markers[v], ls="", label=v) for v in variants
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper right", ncol=2)
    ax.set_xlabel(f"agents per GPU (`{headline}` SLO, agent workload, prefix caching on)")
    ax.set_ylabel("tool-calling accuracy (%)")
    ax.set_title("Capacity against accuracy (x offset per engine to avoid overlap)", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def build_report(
    results_dir: Path,
    reports_dir: Path,
    config_dir: Path = config.DEFAULT_CONFIG_DIR,
    a40_usd_per_hour: float | None = None,
) -> list[Path]:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    slo_cfg = config.load("slo", config_dir)
    matrix_cfg = config.load("matrix", config_dir)
    qcfg = config.load("quality", config_dir)
    # `fixed` runs are the AIPerf cross-check input, not results
    summaries = [s for s in load_summaries(results_dir) if s["workload"] != "fixed"]
    hourly = hourly_slice_usd(matrix_cfg, a40_usd_per_hour)
    written: list[Path] = []

    if summaries:
        csv = reports_dir / "runs.csv"
        pd.DataFrame(summaries).sort_values(
            ["engine", "variant", "workload", "prefix_cache", "concurrency", "repeat"]
        ).to_csv(csv, index=False)
        written.append(csv)
    for variant in sorted({s["variant"] for s in summaries}):
        p = plot_latency(summaries, slo_cfg, variant, reports_dir / f"latency_{variant}.png")
        if p:
            written.append(p)
    quality = latest_quality(results_dir)
    pareto = plot_pareto(summaries, slo_cfg, quality, reports_dir / "pareto_accuracy_capacity.png")
    if pareto:
        written.append(pareto)
    for fig in (
        plot_caching_hero(summaries, slo_cfg, hourly, reports_dir / "hero_prefix_caching.png"),
        plot_model_size(summaries, slo_cfg, quality, reports_dir / "model_size.png"),
    ):
        if fig:
            written.append(fig)

    digests = sorted(
        {f"{s['engine']}: {s['image_digest']}" for s in summaries if s["image_digest"]}
    )
    dates = sorted(s["date"][:10] for s in summaries)
    price_note = (
        "no GPU price set (matrix.yaml `cost.a40_usd_per_hour`), so costs are n/a"
        if hourly is None
        else f"slice price {hourly:.3f} USD/hour"
    )
    parts = [
        "# Results\n",
        f"Generated by `apg report` on {dt.datetime.now(dt.UTC):%Y-%m-%d} from "
        f"{len(summaries)} run file(s) in `results/raw/`"
        + (f" dated {dates[0]} to {dates[-1]}" if dates else "")
        + f". {price_note}.\n",
        "## Capacity\n",
        capacity_section(summaries, slo_cfg, hourly),
        "## Test matrix\n",
        matrix_section(summaries, slo_cfg, matrix_cfg, read_cells(results_dir)),
        "## Tool-calling accuracy\n",
        quality_section(results_dir, qcfg),
        "## Run-to-run noise\n",
        noise_section(summaries),
        "## Container images\n",
        "\n".join(f"- {d}" for d in digests) + "\n" if digests else "_None recorded._\n",
        "## Plots\n",
        "\n".join(f"![{p.stem}]({p.name})" for p in written if p.suffix == ".png") + "\n",
    ]
    md = reports_dir / "results.md"
    md.write_text("\n".join(parts))
    written.append(md)
    return written
