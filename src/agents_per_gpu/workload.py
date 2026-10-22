"""Seeded workload generation and request building.

Text is built from a fixed word list and sized with the model's own tokenizer, so the token
counts recorded in the workload files are measured, not estimated. Sessions are independent:
session i depends only on (seed, kind, i), so a longer file extends a shorter one.
"""

import json
import random
from pathlib import Path
from typing import Any, Protocol

WORDS = (
    "account address amount archive batch branch budget cache calendar channel client cluster "
    "column command config contact context counter customer dashboard dataset deadline default "
    "delivery device digest document domain driver editor email endpoint entry error event "
    "export field filter folder format gateway group handle header history index input invoice "
    "item journal kernel label ledger limit listing locale lookup mailbox manager margin member "
    "message metric module monitor network note offset order output package page parser payload "
    "payment pipeline policy portal profile project queue quota record region release report "
    "request resource result review route sample schedule schema scope search section segment "
    "server session setting shipment source status storage stream summary support table target "
    "task template ticket token tracker update upload usage user vendor version view widget "
    "window worker workflow alpha bravo delta energy forest garden harbor island jungle meadow "
    "nectar orchid pebble quartz river summit timber valley willow yonder zephyr"
).split()
VERBS = ("get", "list", "search", "create", "update", "delete", "send", "fetch", "check", "find")
NOUNS = ("order", "user", "invoice", "ticket", "file", "event", "record", "report", "device")


class Tokenizer(Protocol):
    def count(self, text: str) -> int: ...


class HFTokenizer:
    """tokenizers-library wrapper. `name` is a local tokenizer.json path or a Hub id."""

    def __init__(self, name: str):
        from tokenizers import Tokenizer as _Tokenizer

        if Path(name).is_file():
            self._tok = _Tokenizer.from_file(name)
        else:
            self._tok = _Tokenizer.from_pretrained(name)

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False).ids)


def make_text(tok: Tokenizer, rng: random.Random, n: int) -> tuple[str, int]:
    """Text of about n tokens. Returns (text, measured token count)."""
    if n <= 0:
        return "", 0
    words = [rng.choice(WORDS) for _ in range(n)]
    lo, hi = 0, n  # each word is at least one token, so n words never undershoot
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if tok.count(" ".join(words[:mid])) <= n:
            lo = mid
        else:
            hi = mid - 1
    text = " ".join(words[:lo])
    have = tok.count(text)
    if have < n:
        text += " the" * (n - have)
        have = tok.count(text)
    return text, have


def build_system_prompt(
    tok: Tokenizer, rng: random.Random, target_tokens: int, n_tools: int
) -> tuple[str, int]:
    tools = []
    for i in range(n_tools):
        params = {}
        for _ in range(rng.randint(2, 4)):
            params[f"{rng.choice(WORDS)}_{len(params)}"] = {
                "type": "string",
                "description": make_text(tok, rng, rng.randint(5, 12))[0],
            }
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": f"{rng.choice(VERBS)}_{rng.choice(NOUNS)}_{i}",
                    "description": make_text(tok, rng, rng.randint(15, 30))[0],
                    "parameters": {
                        "type": "object",
                        "properties": params,
                        "required": list(params)[:1],
                    },
                },
            }
        )
    head = (
        "You are a tool-using agent. Call one tool per step and wait for its result "
        "before continuing.\n# Tools\n" + "\n".join(json.dumps(t) for t in tools) + "\n# Notes\n"
    )
    have = tok.count(head)
    if have > target_tokens:
        raise ValueError(
            f"{n_tools} tool schemas need {have} tokens; raise system_prompt_tokens"
            f" (now {target_tokens}) or lower tools"
        )
    pad, _ = make_text(tok, rng, target_tokens - have)
    text = head + pad
    return text, tok.count(text)


def _agent_session(tok: Tokenizer, cfg: dict[str, Any], seed: int, i: int) -> dict[str, Any]:
    rng = random.Random(f"{seed}:agent:{i}")
    task, task_n = make_text(tok, rng, rng.randint(*cfg["user_task_tokens"]))
    steps = []
    for _ in range(rng.randint(*cfg["steps"])):
        out_n = rng.randint(*cfg["tool_call_output_tokens"])
        call, call_n = make_text(tok, rng, out_n)
        result, result_n = make_text(tok, rng, rng.randint(*cfg["tool_result_tokens"]))
        steps.append(
            {
                "output_tokens": out_n,
                "assistant": call,
                "assistant_tokens": call_n,
                "tool_result": result,
                "tool_result_tokens": result_n,
            }
        )
    return {
        "id": i,
        "user_task": task,
        "user_task_tokens": task_n,
        "steps": steps,
        "final_output_tokens": rng.randint(*cfg["final_answer_tokens"]),
    }


def _single_request(tok: Tokenizer, cfg: dict[str, Any], seed: int, kind: str, i: int) -> dict:
    rng = random.Random(f"{seed}:{kind}:{i}")
    prompt, prompt_n = make_text(tok, rng, rng.randint(*cfg["prompt_tokens"]))
    return {
        "id": i,
        "prompt": prompt,
        "prompt_tokens": prompt_n,
        "output_tokens": rng.randint(*cfg["output_tokens"]),
    }


def generate(
    kind: str,
    cfg: dict[str, Any],
    seed: int,
    tok: Tokenizer,
    tokenizer_name: str,
    count: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return (meta, items). `kind` is a key of workloads.yaml ('agent', 'chat', 'fixed')."""
    meta: dict[str, Any] = {
        "kind": kind,
        "shape": "agent" if "steps" in cfg else "single",
        "seed": seed,
        "tokenizer": tokenizer_name,
        "config": cfg,
    }
    if meta["shape"] == "agent":
        n = count or cfg["sessions"]
        system, system_n = build_system_prompt(
            tok, random.Random(f"{seed}:agent:system"), cfg["system_prompt_tokens"], cfg["tools"]
        )
        meta.update(system_prompt=system, system_prompt_tokens=system_n)
        return meta, [_agent_session(tok, cfg, seed, i) for i in range(n)]
    n = count or cfg["requests"]
    return meta, [_single_request(tok, cfg, seed, kind, i) for i in range(n)]


def write_workload(path: Path, meta: dict[str, Any], items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps({"type": "meta", **meta}) + "\n")
        for item in items:
            f.write(json.dumps({"type": "item", **item}) + "\n")


def load_workload(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta: dict[str, Any] = {}
    items: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.pop("type") == "meta":
                meta = rec
            else:
                items.append(rec)
    return meta, items


def workload_path(workloads_dir: Path, kind: str, seed: int) -> Path:
    return Path(workloads_dir) / f"{kind}-seed{seed}.jsonl"


def build_request(
    meta: dict[str, Any], item: dict[str, Any], step: int, nonce: str
) -> tuple[list[dict[str, str]], int]:
    """Messages and max_tokens for one request.

    `nonce` is unique per (run, session) so repeated runs can't hit the prefix cache on the
    session-specific part of the prompt. It is placed after the shared system prompt, which
    must stay a common prefix for the agent workload.
    """
    if meta["shape"] == "single":
        return [{"role": "user", "content": f"[req {nonce}] {item['prompt']}"}], item[
            "output_tokens"
        ]

    messages = [
        {"role": "system", "content": meta["system_prompt"]},
        {"role": "user", "content": f"[task {nonce}] {item['user_task']}"},
    ]
    steps = item["steps"]
    for done in steps[:step]:
        messages.append({"role": "assistant", "content": done["assistant"]})
        messages.append(
            {
                "role": "user",
                "content": f"<tool_response>\n{done['tool_result']}\n</tool_response>",
            }
        )
    max_tokens = steps[step]["output_tokens"] if step < len(steps) else item["final_output_tokens"]
    return messages, max_tokens


def n_requests(meta: dict[str, Any], item: dict[str, Any]) -> int:
    return len(item["steps"]) + 1 if meta["shape"] == "agent" else 1
