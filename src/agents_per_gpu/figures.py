"""README graphics: the prefix-caching headline and the model-size comparison.

Both are drawn from the same capacity rows as `reports/results.md`, so `apg report`
regenerates them with every other number.
"""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from agents_per_gpu.analysis import add_cost, capacity_rows  # noqa: E402

ENGINES = [("vllm", "vLLM"), ("sglang", "SGLang"), ("trtllm", "TensorRT-LLM")]
# validated together on SURFACE (dark mode, all pairs, colour-blind checks included)
ENGINE_COLORS = {"vllm": "#3987e5", "sglang": "#d95926", "trtllm": "#199e70"}
VARIANTS = [
    ("qwen3-4b-bf16", "Qwen3-4B, BF16"),
    ("qwen3-8b-bf16", "Qwen3-8B, BF16"),
    ("qwen3-8b-int8", "Qwen3-8B, INT8"),
    ("qwen3-8b-int4-awq", "Qwen3-8B, INT4-AWQ"),
]
SURFACE, PANEL, INK, INK2, GRID, OFF = (
    "#151924",
    "#1d2232",
    "#f0efec",
    "#c3c2b7",
    "#2f3650",
    "#8a8f9e",
)


def _canvas(width_px: int, height_px: int, dpi: int = 100):
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi, facecolor=SURFACE)
    return fig


def _by(rows: list[dict[str, Any]], **match: Any) -> dict[str, Any] | None:
    for r in rows:
        if all(r[k] == v for k, v in match.items()):
            return r
    return None


def _usd(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def plot_caching_hero(
    summaries, slo_cfg, hourly, path: Path, size: tuple[int, int] = (1280, 640)
) -> Path | None:
    """Agents per GPU with prefix caching off and on, Qwen3-4B BF16, agent traffic."""
    profile = "interactive"
    rows = [
        add_cost(r, hourly)
        for r in capacity_rows(
            [s for s in summaries if s["workload"] == "agent" and s["variant"] == "qwen3-4b-bf16"],
            slo_cfg,
            profile,
        )
    ]
    pairs = []
    for engine, label in ENGINES:
        off = _by(rows, engine=engine, prefix_cache=False)
        on = _by(rows, engine=engine, prefix_cache=True)
        if off is None or on is None:
            return None
        pairs.append((label, off, on))

    w, h = size
    tall = h / w > 0.55
    fig = _canvas(w, h)
    left, right = 0.06, 0.95
    fig.text(
        left, 0.90, "How many AI agents fit on one GPU?", color=INK, fontsize=30, weight="bold"
    )
    fig.text(
        left,
        0.835,
        "Qwen3-4B on one NVIDIA A40, agent traffic, p95 TTFT ≤ 1 s and ITL ≤ 50 ms",
        color=INK2,
        fontsize=15,
    )
    top = 0.68 if tall else 0.70
    ax = fig.add_axes((0.21, 0.22, 0.50 if not tall else 0.48, top - 0.22), facecolor=SURFACE)
    xmax = max(max(off["capacity"], on["capacity"]) for _, off, on in pairs) + 1.2
    ax.set_xlim(0, xmax)
    ax.set_ylim(len(pairs) - 0.4, -0.6)
    for i, (label, off, on) in enumerate(pairs):
        a, b = off["capacity"], on["capacity"]
        ax.plot([a, b], [i, i], color=OFF, lw=3, solid_capstyle="round", zorder=1)
        ax.annotate(
            "",
            xy=(b - 0.18, i),
            xytext=(a + 0.18, i),
            arrowprops=dict(arrowstyle="-|>", color=OFF, lw=0, mutation_scale=18),
            zorder=2,
        )
        ax.scatter([a], [i], s=230, facecolor=SURFACE, edgecolor=OFF, linewidth=2.5, zorder=3)
        ax.scatter([b], [i], s=260, color=ENGINE_COLORS[ENGINES[i][0]], zorder=3)
        ax.text(a, i - 0.32, str(a), color=INK2, fontsize=14, ha="center")
        ax.text(b, i - 0.32, str(b), color=INK, fontsize=16, ha="center", weight="bold")
        ax.text(-0.35, i, label, color=INK, fontsize=17, ha="right", va="center")
    ax.set_xticks(range(0, int(xmax) + 1))
    ax.tick_params(colors=INK2, labelsize=12, length=0)
    ax.set_yticks([])
    ax.grid(axis="x", color=GRID, lw=1)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel("concurrent agent sessions within the SLO", color=INK2, fontsize=13)

    # cost per task, caching off -> on
    cx = 0.76
    fig.text(cx, top + 0.035, "cost per agent task", color=INK2, fontsize=13)
    for i, (_, off, on) in enumerate(pairs):
        y = ax.transData.transform((0, i))
        _, fy = fig.transFigure.inverted().transform(y)
        fig.text(cx, fy, _usd(off["usd_per_task"]), color=INK2, fontsize=15, va="center")
        fig.text(cx + 0.085, fy, "→", color=OFF, fontsize=15, va="center")
        fig.text(
            cx + 0.115,
            fy,
            _usd(on["usd_per_task"]),
            color=INK,
            fontsize=15,
            va="center",
            weight="bold",
        )

    # legend in words and marks, so the encoding never depends on colour alone
    ly = 0.045
    fig.text(left, ly, "○  prefix caching off      ●  prefix caching on", color=INK2, fontsize=13)
    fig.text(
        right,
        ly,
        "worst of 3–6 runs · measured 2026-10-21/22",
        color=INK2,
        fontsize=12,
        ha="right",
    )
    fig.savefig(path, dpi=fig.dpi, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_model_size(
    summaries, slo_cfg, quality, path: Path, size: tuple[int, int] = (1280, 640)
) -> Path | None:
    """Agents per GPU at the 8 s step SLO by model variant and engine, with accuracy."""
    profile = "agent-step-relaxed"
    rows = capacity_rows(
        [s for s in summaries if s["workload"] == "agent" and s["prefix_cache"]],
        slo_cfg,
        profile,
    )
    if not rows:
        return None
    w, h = size
    fig = _canvas(w, h)
    fig.text(
        0.06,
        0.90,
        "A quantized 8B serves more agents than a 4B",
        color=INK,
        fontsize=28,
        weight="bold",
    )
    fig.text(
        0.06,
        0.835,
        "Agents per A40 at p95 step latency ≤ 8 s, prefix caching on, with tool-calling accuracy",
        color=INK2,
        fontsize=15,
    )
    ax = fig.add_axes((0.25, 0.19, 0.50, 0.57), facecolor=SURFACE)
    bar_h = 0.22
    xmax = 7
    for vi, (variant, vlabel) in enumerate(VARIANTS):
        accs = []
        for ei, (engine, _) in enumerate(ENGINES):
            y = vi + (ei - 1) * (bar_h + 0.05)
            r = _by(rows, engine=engine, variant=variant)
            q = quality.get((engine, variant))
            if q:
                accs.append(q["accuracy"] * 100)
            if r is None or r.get("capacity") is None:
                ax.text(
                    0.1, y, "failed to load", color=INK2, fontsize=11, va="center", style="italic"
                )
                continue
            cap = r["capacity"]
            if cap:
                ax.barh(y, cap, height=bar_h, color=ENGINE_COLORS[engine], zorder=2)
            own = engine == "trtllm" and q and q.get("quant_path")
            ax.text(
                cap + 0.12, y, f"{cap}{' †' if own else ''}", color=INK, fontsize=12, va="center"
            )
        ax.text(-0.25, vi, vlabel, color=INK, fontsize=15, ha="right", va="center")
        if accs:
            lo, hi = min(accs), max(accs)
            acc = f"{lo:.1f} %" if lo == hi else f"{lo:.1f}–{hi:.1f} %"
            ax.text(xmax + 0.6, vi, acc, color=INK, fontsize=15, va="center")
    ax.text(xmax + 0.6, -0.75, "accuracy", color=INK2, fontsize=13, va="center")
    ax.set_xlim(0, xmax)
    ax.set_ylim(len(VARIANTS) - 0.5, -0.9)
    ax.set_xticks(range(0, xmax + 1))
    ax.set_yticks([])
    ax.tick_params(colors=INK2, labelsize=12, length=0)
    ax.grid(axis="x", color=GRID, lw=1, zorder=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel("concurrent agent sessions within the SLO", color=INK2, fontsize=13)
    # legend: colour plus the engine's name, in a fixed order
    x = 0.06
    for engine, label in ENGINES:
        fig.patches.append(
            plt.Rectangle(
                (x, 0.035), 0.014, 0.028, transform=fig.transFigure, color=ENGINE_COLORS[engine]
            )
        )
        fig.text(x + 0.02, 0.04, label, color=INK2, fontsize=13)
        x += 0.02 + len(label) * 0.0085 + 0.03
    fig.text(
        0.95,
        0.04,
        "† own INT4-AWQ checkpoint · 8B BF16 is too slow for an 8 s step",
        color=INK2,
        fontsize=12,
        ha="right",
    )
    fig.savefig(path, dpi=fig.dpi, facecolor=SURFACE)
    plt.close(fig)
    return path
