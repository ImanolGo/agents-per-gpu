import asyncio

import httpx
import pytest
from fake_engine import FakeEngine
from fake_tokenizer import WordTokenizer

from agents_per_gpu.client import RequestSpec, RunAborted, run_level, stream_chat
from agents_per_gpu.metrics import summarize
from agents_per_gpu.workload import generate

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
SPEC = RequestSpec("fake-model", 0.0, {"ignore_eos": True})


def _client(engine):
    return httpx.AsyncClient(transport=engine.transport(), base_url="http://fake")


def _summ(run):
    meta = {**run["meta"], "engine": "e", "variant": "v", "prefix_cache": True, "date": "d"}
    return summarize({**run, "meta": meta})


def _run(engine, concurrency=3, **kw):
    meta, items = generate("agent", AGENT, 0, WordTokenizer(), "fake")

    async def go():
        async with _client(engine) as http:
            return await run_level(
                http,
                kw.pop("spec", SPEC),
                meta,
                items,
                concurrency=concurrency,
                repeat=1,
                warmup_requests=2,
                warmup_timeout_s=30,
                measure_seconds=0.4,
            )

    return asyncio.run(go())


def test_stream_chat_records_timings_and_tokens():
    async def go():
        async with _client(FakeEngine(base_delay=0.01)) as http:
            return await stream_chat(http, SPEC, [{"role": "user", "content": "a b c"}], 8)

    r = asyncio.run(go())
    assert r["success"] and r["length_ok"]
    assert r["output_tokens"] == 8 and r["output_tokens_source"] == "usage"
    assert r["prompt_tokens"] == 3
    assert len(r["itl_s"]) == 7
    assert 0.01 <= r["ttft_s"] < r["step_latency_s"]
    assert r["step_latency_s"] >= 8 * 0.01 * 0.9


def test_length_mismatch_is_recorded_when_ignore_eos_is_not_honoured():
    async def go():
        async with _client(FakeEngine(honor_ignore_eos=False)) as http:
            return await stream_chat(http, SPEC, [{"role": "user", "content": "x"}], 8)

    r = asyncio.run(go())
    assert r["success"] and not r["length_ok"] and r["output_tokens"] == 7


def test_http_error_is_a_failed_request():
    async def go():
        async with _client(FakeEngine(status=500)) as http:
            return await stream_chat(http, SPEC, [{"role": "user", "content": "x"}], 8)

    r = asyncio.run(go())
    assert not r["success"] and r["error"].startswith("http 500")
    assert r["ttft_s"] is None


def test_run_level_phases_sessions_and_extra_body():
    engine = FakeEngine()
    run = _run(engine)
    phases = {r["phase"] for r in run["requests"]}
    assert {"warmup", "measure"} <= phases
    assert sum(r["phase"] == "warmup" for r in run["requests"]) >= 3  # max(warmup, concurrency)
    assert run["meta"]["window_end"] - run["meta"]["window_start"] == pytest.approx(0.4)
    complete = [s for s in run["sessions"] if s["phase"] == "measure" and s["complete"]]
    assert complete
    assert all(engine_body["ignore_eos"] for engine_body in engine.requests)
    s = _summ(run)
    assert s["error_rate"] == 0 and s["length_mismatch_rate"] == 0
    assert s["output_tok_s"] > 0


def test_workers_run_distinct_sessions_with_unique_nonces():
    engine = FakeEngine()
    _run(engine, concurrency=4)
    firsts = {
        r["messages"][1]["content"].split("]")[0]
        for r in engine.requests
        if len(r["messages"]) == 2
    }
    assert len(firsts) >= 4


def test_dead_server_aborts_instead_of_spinning():
    with pytest.raises(RunAborted):
        _run(FakeEngine(status=500))


def test_latency_grows_with_concurrency():
    p95 = {c: _summ(_run(FakeEngine(per_inflight=0.003), c))["step_p95_s"] for c in (1, 8)}
    assert p95[8] > p95[1]
