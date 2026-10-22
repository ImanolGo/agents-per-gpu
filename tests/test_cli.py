import pytest
from fake_tokenizer import WordTokenizer

from agents_per_gpu import cli, workload
from agents_per_gpu.cli import main


def test_run_requires_engine_variant_and_concurrency():
    with pytest.raises(SystemExit):
        main(["run", "--engine", "vllm"])


def test_workload_command_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(workload, "HFTokenizer", lambda name: WordTokenizer())
    out = tmp_path / "w"
    assert main(["workload", "--profile", "chat", "--count", "3", "--workloads-dir", str(out)]) == 0
    assert (out / "chat-seed0.jsonl").exists()
    assert workload.load_workload(out / "chat-seed0.jsonl")[1][0]["prompt_tokens"] >= 500


def test_workload_unknown_profile(capsys):
    assert main(["workload", "--profile", "nope"]) == 2
    assert "unknown profile" in capsys.readouterr().err


def test_every_command_is_registered():
    help_text = cli.build_parser().format_help()
    for name in ("workload", "run", "sweep", "quality", "report", "plan", "cell"):
        assert name in help_text
