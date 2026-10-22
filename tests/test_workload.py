import random

import pytest
from fake_tokenizer import WordTokenizer

from agents_per_gpu.workload import (
    build_request,
    build_system_prompt,
    generate,
    load_workload,
    make_text,
    n_requests,
    write_workload,
)

TOK = WordTokenizer()
AGENT = {
    "sessions": 6,
    "system_prompt_tokens": 600,
    "tools": 5,
    "user_task_tokens": [5, 10],
    "steps": [3, 5],
    "tool_call_output_tokens": [4, 8],
    "tool_result_tokens": [10, 20],
    "final_answer_tokens": [6, 9],
}
CHAT = {"requests": 4, "prompt_tokens": [20, 40], "output_tokens": [5, 9]}


@pytest.mark.parametrize("n", [1, 7, 100, 513])
def test_make_text_hits_exact_token_count(n):
    text, have = make_text(TOK, random.Random(0), n)
    assert have == n == TOK.count(text)


def test_make_text_is_deterministic():
    assert make_text(TOK, random.Random(3), 50) == make_text(TOK, random.Random(3), 50)


def test_system_prompt_size_and_error():
    text, n = build_system_prompt(TOK, random.Random(0), 600, 5)
    assert n == 600 == TOK.count(text)
    with pytest.raises(ValueError, match="raise system_prompt_tokens"):
        build_system_prompt(TOK, random.Random(0), 20, 5)


def test_agent_workload_ranges_and_determinism():
    meta, items = generate("agent", AGENT, 1, TOK, "fake")
    meta2, items2 = generate("agent", AGENT, 1, TOK, "fake")
    assert (meta, items) == (meta2, items2)
    assert len(items) == 6
    assert meta["system_prompt_tokens"] == 600
    for s in items:
        assert 3 <= len(s["steps"]) <= 5
        assert 5 <= s["user_task_tokens"] <= 10
        assert 6 <= s["final_output_tokens"] <= 9
        for st in s["steps"]:
            assert 4 <= st["output_tokens"] <= 8
            assert st["assistant_tokens"] == st["output_tokens"]
            assert 10 <= st["tool_result_tokens"] <= 20


def test_seed_changes_content_and_sessions_are_independent():
    _, a = generate("agent", AGENT, 1, TOK, "fake")
    _, b = generate("agent", AGENT, 2, TOK, "fake")
    _, longer = generate("agent", AGENT, 1, TOK, "fake", count=9)
    assert a[0]["user_task"] != b[0]["user_task"]
    assert longer[:6] == a


def test_chat_workload():
    meta, items = generate("chat", CHAT, 0, TOK, "fake")
    assert meta["shape"] == "single"
    assert len(items) == 4
    assert all(20 <= i["prompt_tokens"] <= 40 and 5 <= i["output_tokens"] <= 9 for i in items)


def test_roundtrip(tmp_path):
    meta, items = generate("agent", AGENT, 0, TOK, "fake")
    path = tmp_path / "w.jsonl"
    write_workload(path, meta, items)
    assert load_workload(path) == (meta, items)


def test_agent_context_grows_and_system_prompt_stays_shared():
    meta, items = generate("agent", AGENT, 0, TOK, "fake")
    s = items[0]
    sizes, systems = [], set()
    for step in range(n_requests(meta, s)):
        messages, max_tokens = build_request(meta, s, step, "n1")
        sizes.append(sum(TOK.count(m["content"]) for m in messages))
        systems.add(messages[0]["content"])
        assert max_tokens > 0
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)
    assert len(systems) == 1
    assert build_request(meta, s, len(s["steps"]), "n1")[1] == s["final_output_tokens"]
    assert n_requests(meta, s) == len(s["steps"]) + 1


def test_nonce_changes_prompt_after_shared_prefix_only():
    meta, items = generate("agent", AGENT, 0, TOK, "fake")
    a, _ = build_request(meta, items[0], 0, "a")
    b, _ = build_request(meta, items[0], 0, "b")
    assert a[0] == b[0] and a[1] != b[1]


def test_single_request_building():
    meta, items = generate("chat", CHAT, 0, TOK, "fake")
    messages, max_tokens = build_request(meta, items[0], 0, "x")
    assert len(messages) == 1 and "[req x]" in messages[0]["content"]
    assert max_tokens == items[0]["output_tokens"]
