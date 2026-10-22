import math

import pytest
from helpers import make_run

from agents_per_gpu.metrics import capacity, cv, meets_slo, slo_ratio, summarize

AGENT_STEP = {"step_p95_s": 3.0}
INTERACTIVE = {"ttft_p95_s": 1.0, "itl_p95_ms": 50}


def test_summarize_basic():
    run = make_run(step_latencies=[1.0, 2.0, 3.0, 4.0], output_tokens=100, window=10, sessions=2)
    s = summarize(run)
    assert s["n_requests"] == 4
    assert s["error_rate"] == 0
    assert s["output_tok_s"] == pytest.approx(40.0)  # 400 tokens in a 10 s window
    assert s["sessions_per_hour"] == pytest.approx(720.0)
    assert s["step_p50_s"] == pytest.approx(2.5)
    assert s["step_p95_s"] == pytest.approx(3.85)
    assert s["itl_p95_ms"] == pytest.approx(20.0)


def test_summarize_ignores_warmup_and_counts_failures():
    s = summarize(make_run(step_latencies=[1.0] * 9, failures=1))
    assert s["n_requests"] == 10
    assert s["error_rate"] == pytest.approx(0.1)


def test_percentile_is_nan_without_data():
    s = summarize(make_run(step_latencies=[], failures=3))
    assert math.isnan(s["step_p95_s"])
    assert s["error_rate"] == 1.0


def test_slo_ratio_and_check():
    s = summarize(make_run(step_latencies=[1.5] * 10, ttfts=[0.5] * 10, itl=0.06))
    assert slo_ratio(s, AGENT_STEP) == pytest.approx(0.5)
    assert slo_ratio(s, INTERACTIVE) == pytest.approx(1.2)  # ITL 60 ms vs 50 ms limit
    assert meets_slo(s, AGENT_STEP, 0.01)
    assert not meets_slo(s, INTERACTIVE, 0.01)


def test_error_rate_breaks_slo_even_when_latency_is_fine():
    s = summarize(make_run(step_latencies=[1.0] * 9, failures=1))
    assert not meets_slo(s, AGENT_STEP, 0.01)
    assert meets_slo(s, AGENT_STEP, 0.2)


def test_missing_latency_counts_as_broken():
    s = summarize(make_run(step_latencies=[], failures=5))
    assert slo_ratio(s, AGENT_STEP) == math.inf


def _ladder(repeat, step_by_conc):
    return [
        summarize(make_run(concurrency=c, repeat=repeat, step_latencies=[lat] * 10))
        for c, lat in step_by_conc.items()
    ]


def test_capacity_is_last_pass_before_first_failure():
    runs = _ladder(1, {1: 1.0, 2: 2.0, 4: 4.0, 8: 2.0})  # 8 passes again but 4 failed
    cap = capacity(runs, AGENT_STEP, 0.01)
    assert cap["capacity"] == 2
    assert not cap["censored"]


def test_capacity_is_worst_of_repeats():
    runs = _ladder(1, {1: 1.0, 2: 2.0, 4: 2.5}) + _ladder(2, {1: 1.0, 2: 2.0, 4: 3.5})
    cap = capacity(runs, AGENT_STEP, 0.01)
    assert cap["capacity"] == 2
    assert cap["per_repeat"] == {1: 4, 2: 2}
    assert cap["repeats"] == 2


def test_capacity_censored_when_nothing_fails():
    cap = capacity(_ladder(1, {1: 1.0, 2: 1.0}), AGENT_STEP, 0.01)
    assert cap["capacity"] == 2
    assert cap["censored"]


def test_capacity_zero_when_lowest_level_fails():
    cap = capacity(_ladder(1, {1: 5.0, 2: 6.0}), AGENT_STEP, 0.01)
    assert cap["capacity"] == 0
    assert cap["sessions_per_hour"] is None


def test_capacity_cost_inputs_use_minimum_across_repeats():
    a = summarize(make_run(concurrency=4, repeat=1, step_latencies=[1.0] * 10, sessions=10))
    b = summarize(make_run(concurrency=4, repeat=2, step_latencies=[1.0] * 10, sessions=6))
    cap = capacity([a, b], AGENT_STEP, 0.01)
    assert cap["sessions_per_hour"] == pytest.approx(6 / 10 * 3600)


def test_cv():
    assert cv([10, 10, 10]) == 0
    assert cv([9, 10, 11]) == pytest.approx(0.1)
    assert math.isnan(cv([5]))
