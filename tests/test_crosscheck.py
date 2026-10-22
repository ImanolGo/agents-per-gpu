import json

import pytest
from helpers import write_ladder

from agents_per_gpu.cli import main
from agents_per_gpu.crosscheck import client_numbers, compare, render


def _run_file(tmp_path):
    write_ladder(tmp_path, {1: 1.0}, repeats=1, output_tokens=100)
    return next((tmp_path / "raw").rglob("*.jsonl"))


def test_client_numbers(tmp_path):
    n = client_numbers(_run_file(tmp_path))
    assert n["ttft_mean_s"] == pytest.approx(0.2)
    assert n["output_tok_s"] == pytest.approx(200.0)  # 20 x 100 tokens over a 10 s window


def test_compare_converts_ms_and_flags_outliers(tmp_path):
    export = tmp_path / "aiperf.json"
    export.write_text(
        json.dumps(
            {
                "time_to_first_token": {"unit": "ms", "avg": 204.0},
                "output_token_throughput": {"unit": "tokens/sec", "avg": 150.0},
            }
        )
    )
    rows = compare(client_numbers(_run_file(tmp_path)), export)
    ttft, tput = rows
    assert ttft["within_tolerance"] and ttft["diff"] == pytest.approx(-0.0196, abs=1e-3)
    assert not tput["within_tolerance"]
    assert "NO" in render(rows)


def test_missing_aiperf_metric_is_not_a_pass(tmp_path):
    export = tmp_path / "aiperf.json"
    export.write_text("{}")
    rows = compare(client_numbers(_run_file(tmp_path)), export)
    assert all(not r["within_tolerance"] and r["aiperf"] is None for r in rows)


def test_crosscheck_command_exit_code(tmp_path, capsys):
    run = _run_file(tmp_path)
    good = tmp_path / "good.json"
    good.write_text(
        json.dumps(
            {
                "time_to_first_token": {"unit": "ms", "avg": 200},
                "output_token_throughput": {"avg": 200},
            }
        )
    )
    assert main(["crosscheck", "--run", str(run), "--aiperf", str(good)]) == 0
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    assert main(["crosscheck", "--run", str(run), "--aiperf", str(bad)]) == 1
