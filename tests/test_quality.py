import asyncio
import json

import httpx
import pytest

from agents_per_gpu import quality
from agents_per_gpu.quality import (
    accuracy,
    check_call,
    convert_cases,
    extract_calls,
    score_message,
    to_openai_tool,
)

EXPECTED = {
    "name": "calculate_triangle_area",
    "arguments": {"base": [10], "height": [5], "unit": ["units", ""]},
}


def msg(name, args):
    return {"tool_calls": [{"function": {"name": name, "arguments": json.dumps(args)}}]}


def test_correct_call_and_optional_argument_omitted():
    r = score_message(msg("calculate_triangle_area", {"base": 10, "height": 5}), EXPECTED)
    assert r["correct"] and r["source"] == "tool_calls"


def test_numbers_compare_across_int_and_float():
    assert score_message(msg("calculate_triangle_area", {"base": 10.0, "height": 5}), EXPECTED)[
        "correct"
    ]


def test_wrong_value_missing_and_extra_arguments_fail():
    f = "calculate_triangle_area"
    assert "wrong value" in score_message(msg(f, {"base": 11, "height": 5}), EXPECTED)["reason"]
    assert "missing argument" in score_message(msg(f, {"base": 10}), EXPECTED)["reason"]
    assert (
        "unexpected" in score_message(msg(f, {"base": 10, "height": 5, "x": 1}), EXPECTED)["reason"]
    )
    assert (
        "wrong function"
        in score_message(msg("other", {"base": 10, "height": 5}), EXPECTED)["reason"]
    )


def test_string_normalisation_follows_bfcl():
    exp = {"name": "f", "arguments": {"city": ["San Francisco"]}}
    assert check_call({"name": "f", "arguments": {"city": "san-francisco"}}, exp)[0]
    assert not check_call({"name": "f", "arguments": {"city": "Boston"}}, exp)[0]


def test_nested_values():
    exp = {"name": "f", "arguments": {"xs": [[1, 2, 3]], "o": [{"a": "X"}]}}
    assert check_call({"name": "f", "arguments": {"xs": [1.0, 2, 3], "o": {"a": "x"}}}, exp)[0]
    assert not check_call({"name": "f", "arguments": {"xs": [1, 2], "o": {"a": "x"}}}, exp)[0]


def test_zero_or_several_calls_fail():
    assert not score_message({"content": "no tools"}, EXPECTED)["correct"]
    two = {"tool_calls": msg("a", {})["tool_calls"] * 2}
    assert score_message(two, EXPECTED)["reason"] == "2 tool calls"


def test_fallback_parses_tool_call_tags_from_content():
    content = (
        '<tool_call>\n{"name": "calculate_triangle_area", '
        '"arguments": {"base": 10, "height": 5}}\n</tool_call>'
    )
    calls, source = extract_calls({"content": content})
    assert source == "content" and calls[0]["arguments"] == {"base": 10, "height": 5}
    assert score_message({"content": content}, EXPECTED)["correct"]


def test_malformed_content_call_is_a_failure_not_a_crash():
    assert not score_message({"content": "<tool_call>{bad}</tool_call>"}, EXPECTED)["correct"]


def test_schema_conversion_to_openai_format():
    tool = to_openai_tool(
        {
            "name": "math.factorial",
            "description": "d",
            "parameters": {
                "type": "dict",
                "properties": {
                    "n": {"type": "integer"},
                    "x": {"type": "float"},
                    "t": {"type": "tuple", "items": {"type": "any"}},
                },
                "required": ["n"],
            },
        }
    )
    fn = tool["function"]
    assert fn["name"] == "math_factorial"
    assert fn["parameters"]["type"] == "object"
    props = fn["parameters"]["properties"]
    assert props["x"]["type"] == "number" and props["t"]["type"] == "array"
    assert props["t"]["items"] == {}


def test_convert_cases_is_seeded_and_joins_ground_truth():
    qs = [
        {
            "id": f"simple_{i}",
            "question": [[{"role": "user", "content": "q"}]],
            "function": [{"name": "a.b", "description": "", "parameters": {"type": "dict"}}],
        }
        for i in range(10)
    ]
    ans = [{"id": q["id"], "ground_truth": [{"a.b": {"p": [i]}}]} for i, q in enumerate(qs)]
    a = convert_cases("simple", qs, ans, 4, 0)
    assert a == convert_cases("simple", qs, ans, 4, 0)
    assert len(a) == 4 and a[0]["expected"]["name"] == "a_b"
    assert a != convert_cases("simple", qs, ans, 4, 1)


def test_evaluate_against_fake_endpoint_and_persistence(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "ignore_eos" not in body and body["tools"]
        good = body["messages"][0]["content"] == "good"
        args = {"base": 10, "height": 5} if good else {"base": 1, "height": 5}
        message = msg("calculate_triangle_area", args)
        return httpx.Response(200, json={"choices": [{"message": message}]})

    cases = [
        {
            "id": f"c{i}",
            "category": "simple",
            "messages": [{"role": "user", "content": "good" if i < 3 else "bad"}],
            "tools": [{}],
            "expected": EXPECTED,
        }
        for i in range(4)
    ]
    results = asyncio.run(
        quality.evaluate("http://x", "m", cases, {}, transport=httpx.MockTransport(handler))
    )
    assert accuracy(results) == {"n": 4, "accuracy": 0.75, "simple": 0.75}

    path = tmp_path / "q.jsonl"
    quality.write_quality(path, {"engine": "vllm"}, results)
    meta, loaded = quality.load_quality(path)
    assert meta["accuracy"] == 0.75 and meta["engine"] == "vllm" and len(loaded) == 4
    with pytest.raises(FileExistsError):
        quality.write_quality(path, {}, results)


def test_http_errors_count_as_wrong_answers():
    transport = httpx.MockTransport(lambda r: httpx.Response(400, text="no tools"))
    cases = [{"id": "c", "category": "simple", "messages": [], "tools": [], "expected": EXPECTED}]
    (r,) = asyncio.run(quality.evaluate("http://x", "m", cases, {}, transport=transport))
    assert not r["correct"] and r["reason"].startswith("error")
