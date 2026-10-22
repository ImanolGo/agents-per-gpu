"""`apg` command line. One subcommand per step; each lands in the PLAN.md phase noted."""

import argparse
import sys
from pathlib import Path

from agents_per_gpu import config


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config-dir", type=Path, default=config.DEFAULT_CONFIG_DIR)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--workloads-dir", type=Path, default=Path("workloads"))


def _target_args(p: argparse.ArgumentParser) -> None:
    _common(p)
    p.add_argument("--engine", required=True, choices=["trtllm", "vllm", "sglang"])
    p.add_argument("--variant", required=True)
    p.add_argument("--workload", default="agent", help="key of configs/workloads.yaml")
    p.add_argument(
        "--prefix-cache",
        choices=["on", "off"],
        default="on",
        help="must match how the server was launched (PREFIX_CACHE in servers/*.sh)",
    )
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--image", help="container image the engine runs in; its digest is recorded")
    p.add_argument("--engine-version", help="engine package version, when not run from a container")
    p.add_argument("--quant-path", help="how the quantized checkpoint was made, if not in config")
    p.add_argument("--measure-seconds", type=float, help="override matrix.yaml")


def _target(args: argparse.Namespace):
    from agents_per_gpu.sweep import Target

    return Target(
        engine=args.engine,
        variant=args.variant,
        workload=args.workload,
        prefix_cache=args.prefix_cache == "on",
        base_url=args.url,
        image=args.image,
        engine_version=args.engine_version,
        quant_path=args.quant_path,
        results_dir=args.results_dir,
        workloads_dir=args.workloads_dir,
        config_dir=args.config_dir,
    )


def _cmd_workload(args: argparse.Namespace) -> int:
    from agents_per_gpu.workload import HFTokenizer, generate, workload_path, write_workload

    workloads = config.load("workloads", args.config_dir)
    models = config.load("models", args.config_dir)
    seed = workloads["seed"] if args.seed is None else args.seed
    if args.profile not in workloads or args.profile == "seed":
        print(f"unknown profile {args.profile!r}", file=sys.stderr)
        return 2
    tokenizer = args.tokenizer or models["tokenizer"]
    meta, items = generate(
        args.profile, workloads[args.profile], seed, HFTokenizer(tokenizer), tokenizer, args.count
    )
    path = workload_path(args.workloads_dir, args.profile, seed)
    write_workload(path, meta, items)
    print(f"wrote {len(items)} items to {path}", file=sys.stderr)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from agents_per_gpu.client import RunAborted
    from agents_per_gpu.sweep import Context, execute_run

    try:
        execute_run(Context(_target(args)), args.concurrency, args.repeat, args.measure_seconds)
    except RunAborted as e:
        print(f"aborted: {e}", file=sys.stderr)
        return 1
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    from agents_per_gpu.sweep import sweep

    ladder = [int(c) for c in args.concurrencies.split(",")] if args.concurrencies else None
    return sweep(_target(args), ladder, args.repeats, args.measure_seconds)


def _cmd_quality(args: argparse.Namespace) -> int:
    import asyncio
    import datetime as dt

    from agents_per_gpu import quality
    from agents_per_gpu.sweep import image_digest, served_model

    qcfg = config.load("quality", args.config_dir)
    matrix = config.load("matrix", args.config_dir)
    models = config.load("models", args.config_dir)
    if args.variant not in models["variants"]:
        print(f"unknown variant {args.variant!r}", file=sys.stderr)
        return 2
    variant = models["variants"][args.variant]
    cases_path = args.cases or Path(qcfg["cases"])
    if not cases_path.exists():
        print(
            f"{cases_path} not found; run: uv run python scripts/prepare_bfcl.py", file=sys.stderr
        )
        return 2
    cases = quality.load_cases(cases_path)

    req = matrix["request"]
    extra = {**req["extra_body"], **(req.get("engine_extra_body") or {}).get(args.engine, {})}
    extra.pop("ignore_eos", None)  # would pad every answer to max_tokens
    model = served_model(args.url)
    results = asyncio.run(
        quality.evaluate(args.url, model, cases, extra, qcfg["concurrency"], req["timeout_s"])
    )
    now = dt.datetime.now(dt.UTC)
    meta = {
        "engine": args.engine,
        "variant": args.variant,
        "model": model,
        "hf_id": variant["hf_id"],
        "quant_path": args.quant_path or (variant.get("quant_path") or {}).get(args.engine),
        "image": args.image,
        "image_digest": image_digest(args.image),
        "cases_file": str(cases_path),
        "date": now.isoformat(timespec="seconds"),
    }
    path = quality.result_path(
        args.results_dir, args.engine, args.variant, now.strftime("%Y%m%dT%H%M%SZ")
    )
    quality.write_quality(path, meta, results)
    acc = quality.accuracy(results)
    print(f"accuracy {acc['accuracy']:.1%} on {acc['n']} cases -> {path}", file=sys.stderr)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from agents_per_gpu.report import build_report

    written = build_report(
        args.results_dir, args.reports_dir, args.config_dir, args.a40_usd_per_hour
    )
    for p in written:
        print(p, file=sys.stderr)
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    from agents_per_gpu.analysis import (
        add_cost,
        capacity_rows,
        hourly_slice_usd,
        latest_quality,
        load_summaries,
    )
    from agents_per_gpu.plan import build_plan, render

    slo_cfg = config.load("slo", args.config_dir)
    matrix_cfg = config.load("matrix", args.config_dir)
    args.slo = args.slo or slo_cfg.get("headline_profile", "agent-step")
    if args.slo not in slo_cfg["profiles"]:
        print(
            f"unknown SLO profile {args.slo!r}; have {list(slo_cfg['profiles'])}", file=sys.stderr
        )
        return 2
    prefix_cache = args.prefix_cache == "on"
    summaries = [
        s
        for s in load_summaries(args.results_dir)
        if s["workload"] == args.workload and s["prefix_cache"] == prefix_cache
    ]
    if not summaries:
        print("no measured runs for this workload and prefix-cache setting", file=sys.stderr)
        return 1
    hourly = hourly_slice_usd(matrix_cfg, args.a40_usd_per_hour)
    rows = [add_cost(r, hourly) for r in capacity_rows(summaries, slo_cfg, args.slo)]
    accuracy = {k: m["accuracy"] * 100 for k, m in latest_quality(args.results_dir).items()}
    plan = build_plan(rows, args.users, args.sessions_per_user, hourly, accuracy, args.min_accuracy)
    context = f"SLO {args.slo}, {args.workload} workload, prefix cache {args.prefix_cache}"
    print(render(plan, args.users, args.sessions_per_user, context))
    return 0 if plan["options"] else 1


def _cmd_crosscheck(args: argparse.Namespace) -> int:
    from agents_per_gpu import crosscheck

    rows = crosscheck.compare(crosscheck.client_numbers(args.run), args.aiperf)
    print(crosscheck.render(rows))
    return 0 if all(r["within_tolerance"] for r in rows) else 1


def _cmd_cell(args: argparse.Namespace) -> int:
    import datetime as dt

    from agents_per_gpu.analysis import append_cell

    append_cell(
        args.results_dir,
        {
            "engine": args.engine,
            "variant": args.variant,
            "status": args.status,
            "reason": args.reason,
            "date": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        },
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="apg", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("workload", help="Generate seeded workloads into workloads/")
    _common(p)
    p.add_argument("--profile", default="agent", help="key of configs/workloads.yaml")
    p.add_argument("--seed", type=int, help="default: seed in configs/workloads.yaml")
    p.add_argument("--count", type=int, help="override the number of sessions/requests")
    p.add_argument("--tokenizer", help="default: tokenizer in configs/models.yaml")
    p.set_defaults(func=_cmd_workload)

    p = sub.add_parser("run", help="Run one engine x variant x workload at one concurrency")
    _target_args(p)
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--repeat", type=int, default=1, help="repeat index; names the result file")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("sweep", help="Run the concurrency ladder with repeats")
    _target_args(p)
    p.add_argument("--concurrencies", help="comma list; default: matrix.yaml ladder")
    p.add_argument("--repeats", type=int, help="default: matrix.yaml repeats")
    p.set_defaults(func=_cmd_sweep)

    p = sub.add_parser("quality", help="Score tool-calling accuracy through the served endpoint")
    _common(p)
    p.add_argument("--engine", required=True, choices=["trtllm", "vllm", "sglang"])
    p.add_argument("--variant", required=True)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--image")
    p.add_argument("--quant-path")
    p.add_argument("--cases", type=Path, help="default: cases in configs/quality.yaml")
    p.set_defaults(func=_cmd_quality)

    p = sub.add_parser("report", help="Rebuild every table and plot in reports/ from results/")
    _common(p)
    p.add_argument("--reports-dir", type=Path, default=Path("reports"))
    p.add_argument("--a40-usd-per-hour", type=float, help="full A40 price; default: matrix.yaml")
    p.set_defaults(func=_cmd_report)

    p = sub.add_parser("plan", help="Size a deployment: slices and cost for N users at an SLO")
    _common(p)
    p.add_argument("--users", type=int, required=True)
    p.add_argument(
        "--sessions-per-user", type=float, default=1.0, help="concurrent agent sessions per user"
    )
    p.add_argument("--slo", help="profile in configs/slo.yaml; default: its headline_profile")
    p.add_argument("--workload", default="agent")
    p.add_argument("--prefix-cache", choices=["on", "off"], default="on")
    p.add_argument("--a40-usd-per-hour", type=float, help="full A40 price; default: matrix.yaml")
    p.add_argument("--min-accuracy", type=float, help="exclude variants below this accuracy (%%)")
    p.set_defaults(func=_cmd_plan)

    p = sub.add_parser(
        "crosscheck", help="Compare one apg run with an AIPerf export (TTFT and tok/s within 5%%)"
    )
    p.add_argument("--run", type=Path, required=True, help="raw apg run file")
    p.add_argument("--aiperf", type=Path, required=True, help="profile_export_aiperf.json")
    p.set_defaults(func=_cmd_crosscheck)

    p = sub.add_parser("cell", help="Record that an engine x variant cell could not run, and why")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--engine", required=True, choices=["trtllm", "vllm", "sglang"])
    p.add_argument("--variant", required=True)
    p.add_argument("--status", default="not supported")
    p.add_argument("--reason", required=True, help="the error or the finding, verbatim")
    p.set_defaults(func=_cmd_cell)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
