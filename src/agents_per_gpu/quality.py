"""Tool-calling accuracy on a fixed BFCL subset, scored through the served endpoint.

This is BFCL-style, not the official BFCL checker: one expected call per case, name and
argument values compared after BFCL-like normalisation. Scores are comparable between the
variants and engines here, not with the published leaderboard.
"""

import asyncio
import json
import random
import re
from pathlib import Path
from typing import Any

import httpx

CATEGORIES = {"simple": "BFCL_v4_simple_python", "multiple": "BFCL_v4_multiple"}
_TYPE_MAP = {"dict": "object", "float": "number", "tuple": "array", "bool": "boolean"}
_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def safe_name(name: str) -> str:
    """OpenAI function names can't contain dots; BFCL's own handlers do the same swap."""
    return name.replace(".", "_")


def _convert_schema(node: Any) -> Any:
    if isinstance(node, list):
        return [_convert_schema(n) for n in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for key, value in node.items():
        if key == "type" and isinstance(value, str):
            if value != "any":
                out[key] = _TYPE_MAP.get(value, value)
        else:
            out[key] = _convert_schema(value)
    return out


def to_openai_tool(fn: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": safe_name(fn["name"]),
            "description": fn.get("description", ""),
            "parameters": _convert_schema(fn["parameters"]),
        },
    }


def convert_cases(
    category: str, questions: list[dict], answers: list[dict], n: int, seed: int
) -> list[dict]:
    """Pick n cases of one category (seeded) and join each with its ground truth."""
    truth = {a["id"]: a["ground_truth"] for a in answers}
    chosen = random.Random(f"{seed}:{category}").sample(questions, n)
    cases = []
    for q in sorted(chosen, key=lambda q: q["id"]):
        (gt,) = truth[q["id"]]  # simple and multiple have exactly one expected call
        ((name, params),) = gt.items()
        cases.append(
            {
                "id": q["id"],
                "category": category,
                "messages": q["question"][0],
                "tools": [to_openai_tool(f) for f in q["function"]],
                "expected": {"name": safe_name(name), "arguments": params},
            }
        )
    return cases


def _norm(v: Any) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, int | float):
        return float(v)
    if isinstance(v, str):
        return re.sub(r"[ ,./\-_*^]", "", v.replace("'", '"')).lower()
    if isinstance(v, list | tuple):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def check_call(call: dict[str, Any], expected: dict[str, Any]) -> tuple[bool, str]:
    if call["name"] != expected["name"]:
        return False, f"wrong function {call['name']!r}"
    args, allowed = call["arguments"], expected["arguments"]
    for key in args:
        if key not in allowed:
            return False, f"unexpected argument {key!r}"
    for key, options in allowed.items():
        if key not in args:
            if "" in options:  # "" marks an optional argument
                continue
            return False, f"missing argument {key!r}"
        if not any(_norm(args[key]) == _norm(o) for o in options):
            return False, f"wrong value for {key!r}: {args[key]!r}"
    return True, "ok"


def extract_calls(message: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """Tool calls from a chat message: the server's parser first, then <tool_call> in content."""
    calls, source = [], "tool_calls"
    for tc in message.get("tool_calls") or []:
        fn = tc["function"]
        args = fn.get("arguments") or {}
        calls.append({"name": fn["name"], "arguments": args})
    if not calls:
        source = "content"
        for m in _TOOL_CALL.finditer(message.get("content") or ""):
            try:
                obj = json.loads(m.group(1))
                calls.append({"name": obj["name"], "arguments": obj.get("arguments") or {}})
            except (json.JSONDecodeError, KeyError):
                calls.append({"name": "", "arguments": {}})
    for c in calls:
        if isinstance(c["arguments"], str):
            try:
                c["arguments"] = json.loads(c["arguments"])
            except json.JSONDecodeError:
                c["arguments"] = {"__unparsable__": c["arguments"]}
    return calls, source


def score_message(message: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    calls, source = extract_calls(message)
    if len(calls) != 1:
        return {"correct": False, "reason": f"{len(calls)} tool calls", "source": source}
    ok, reason = check_call(calls[0], expected)
    return {"correct": ok, "reason": reason, "source": source, "call": calls[0]}


def load_cases(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


async def evaluate(
    base_url: str,
    model: str,
    cases: list[dict],
    extra_body: dict[str, Any],
    concurrency: int = 4,
    timeout_s: float = 300,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(
        base_url=base_url, timeout=timeout_s, trust_env=False, transport=transport
    ) as http:

        async def one(case: dict) -> dict[str, Any]:
            payload = {
                "model": model,
                "messages": case["messages"],
                "tools": case["tools"],
                "tool_choice": "auto",
                "temperature": 0,
                "max_tokens": 512,
                **extra_body,
            }
            async with sem:
                try:
                    r = await http.post("/v1/chat/completions", json=payload)
                    r.raise_for_status()
                    result = score_message(r.json()["choices"][0]["message"], case["expected"])
                except (httpx.HTTPError, KeyError, ValueError) as e:
                    result = {"correct": False, "reason": f"error: {type(e).__name__}: {e}"[:200]}
            return {"id": case["id"], "category": case["category"], **result}

        return list(await asyncio.gather(*(one(c) for c in cases)))


def accuracy(results: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(results)}
    out["accuracy"] = sum(r["correct"] for r in results) / len(results) if results else None
    for cat in sorted({r["category"] for r in results}):
        rs = [r for r in results if r["category"] == cat]
        out[cat] = sum(r["correct"] for r in rs) / len(rs)
    return out


def result_path(results_dir: Path, engine: str, variant: str, stamp: str) -> Path:
    return Path(results_dir) / "quality" / engine / f"{variant}-{stamp}.jsonl"


def write_quality(path: Path, meta: dict[str, Any], results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as f:  # raw results are append-only: never replace a file
        f.write(json.dumps({"type": "meta", **meta, **accuracy(results)}) + "\n")
        for r in results:
            f.write(json.dumps({"type": "case", **r}) + "\n")


def load_quality(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta, cases = {}, []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.pop("type") == "meta":
                meta = rec
            else:
                cases.append(rec)
    return meta, cases
