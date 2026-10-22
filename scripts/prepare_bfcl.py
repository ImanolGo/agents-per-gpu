#!/usr/bin/env python3
"""Build data/quality/bfcl_subset.jsonl from the public BFCL repository (Apache-2.0).

    uv run python scripts/prepare_bfcl.py

Downloads BFCL_v4_simple_python and BFCL_v4_multiple (questions + possible answers), samples
`cases_per_category` of each with the seed in configs/quality.yaml and writes normalised cases.
"""

import json
import sys
from pathlib import Path

import httpx

from agents_per_gpu import config
from agents_per_gpu.quality import CATEGORIES, convert_cases

BASE = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/"
    "berkeley-function-call-leaderboard/bfcl_eval/data"
)


def fetch(path: str) -> list[dict]:
    r = httpx.get(f"{BASE}/{path}", follow_redirects=True, timeout=60)
    r.raise_for_status()
    return [json.loads(line) for line in r.text.splitlines() if line.strip()]


def main() -> int:
    cfg = config.load("quality")
    cases = []
    for category, stem in CATEGORIES.items():
        questions, answers = fetch(f"{stem}.json"), fetch(f"possible_answer/{stem}.json")
        cases += convert_cases(category, questions, answers, cfg["cases_per_category"], cfg["seed"])
    out = Path(cfg["cases"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(c) + "\n" for c in cases))
    print(f"wrote {len(cases)} cases to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
