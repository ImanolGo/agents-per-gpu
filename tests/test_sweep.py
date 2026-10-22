import shutil
from pathlib import Path

import httpx
import pytest
import yaml
from fake_engine import FakeEngine
from fake_tokenizer import WordTokenizer

from agents_per_gpu import sweep as sweep_mod
from agents_per_gpu.metrics import load_run, summarize
from agents_per_gpu.sweep import Target, run_path, should_stop, write_run
from agents_per_gpu.workload import generate, workload_path, write_workload

AGENT = {
    "sessions": 4,
    "system_prompt_tokens": 600,
    "tools": 3,
    "user_task_tokens": [5, 8],
    "steps": [2, 3],
    "tool_call_output_tokens": [3, 5],
    "tool_result_tokens": [5, 8],
    "final_answer_tokens": [4, 6],
}


@pytest.fixture
def env(tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    shutil.copytree(Path("configs"), configs)
    matrix = yaml.safe_load((configs / "matrix.yaml").read_text())
    matrix["warmup_requests"] = 2
    (configs / "matrix.yaml").write_text(yaml.safe_dump(matrix))

    workloads = tmp_path / "workloads"
    meta, items = generate("agent", AGENT, 0, WordTokenizer(), "fake")
    write_workload(workload_path(workloads, "agent", 0), meta, items)

    engine = FakeEngine()
    real = httpx.AsyncClient
    monkeypatch.setattr(
        sweep_mod.httpx, "AsyncClient", lambda **kw: real(transport=engine.transport(), **kw)
    )
    monkeypatch.setattr(sweep_mod, "served_model", lambda url: "fake-model")
    monkeypatch.setattr(sweep_mod, "image_digest", lambda image: "sha256:abc" if image else None)
    target = Target(
        "vllm",
        "qwen3-4b-bf16",
        "agent",
        True,
        image="vllm/vllm-openai:test",
        results_dir=tmp_path / "results",
        workloads_dir=workloads,
        config_dir=configs,
    )
    return target, engine


def test_sweep_writes_one_file_per_run_with_full_metadata(env):
    target, _ = env
    assert sweep_mod.sweep(target, concurrencies=[1, 2], repeats=2, measure_seconds=0.3) == 0
    files = sorted((target.results_dir / "raw/vllm/qwen3-4b-bf16").glob("*.jsonl"))
    assert [f.name for f in files] == [
        "agent-pcon-c001-r1.jsonl",
        "agent-pcon-c001-r2.jsonl",
        "agent-pcon-c002-r1.jsonl",
        "agent-pcon-c002-r2.jsonl",
    ]
    meta = load_run(files[0])["meta"]
    for key in (
        "engine",
        "variant",
        "model",
        "quant_path",
        "image",
        "image_digest",
        "workload",
        "workload_seed",
        "concurrency",
        "repeat",
        "prefix_cache",
        "date",
        "slo_profiles",
        "window_start",
        "window_end",
    ):
        assert key in meta
    assert meta["image_digest"] == "sha256:abc"
    assert meta["workload_seed"] == 0
    assert summarize(load_run(files[0]))["n_requests"] > 0


def test_existing_runs_are_skipped_not_overwritten(env):
    target, engine = env
    sweep_mod.sweep(target, concurrencies=[1], repeats=1, measure_seconds=0.2)
    path = run_path(target.results_dir, target, 1, 1)
    before = path.read_text()
    n_requests = len(engine.requests)
    sweep_mod.sweep(target, concurrencies=[1], repeats=1, measure_seconds=0.2)
    assert path.read_text() == before and len(engine.requests) == n_requests


def test_write_run_refuses_to_replace(tmp_path):
    run = {"meta": {"a": 1}, "requests": [], "sessions": []}
    write_run(tmp_path / "x.jsonl", run)
    with pytest.raises(FileExistsError):
        write_run(tmp_path / "x.jsonl", run)


def test_missing_workload_file_gives_actionable_error(env):
    target, _ = env
    target.workloads_dir = target.results_dir / "nope"
    with pytest.raises(SystemExit, match="apg workload"):
        sweep_mod.sweep(target, concurrencies=[1], repeats=1)


def test_unknown_variant(env):
    target, _ = env
    target.variant = "nope"
    with pytest.raises(SystemExit, match="unknown variant"):
        sweep_mod.Context(target)


def test_should_stop_needs_every_profile_broken_by_the_factor():
    profiles = {"interactive": {"ttft_p95_s": 1.0}, "agent-step": {"step_p95_s": 3.0}}
    slow = {"ttft_p95_s": 2.5, "step_p95_s": 7.0}
    mixed = {"ttft_p95_s": 2.5, "step_p95_s": 4.0}
    assert should_stop([slow], profiles, 2.0)
    assert not should_stop([mixed], profiles, 2.0)
    assert should_stop([mixed, slow], profiles, 2.0)  # worst repeat decides
