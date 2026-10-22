# Third-party models and data

Each model and dataset used, with its licence and the date it was checked.

| Item | Source | Licence | Checked |
|---|---|---|---|
| Qwen3-4B | huggingface.co/Qwen/Qwen3-4B | Apache-2.0 (Hub model metadata) | 2026-10-21 |
| Qwen3-8B | huggingface.co/Qwen/Qwen3-8B | Apache-2.0 (Hub model metadata) | 2026-10-21 |
| Qwen3-8B-AWQ | huggingface.co/Qwen/Qwen3-8B-AWQ | Apache-2.0 (Hub model metadata) | 2026-10-21 |
| Qwen3-8B-GPTQ-Int8 | huggingface.co/JunHowie/Qwen3-8B-GPTQ-Int8 (community quantization of Qwen3-8B; no official INT8 exists) | Apache-2.0 (Hub model metadata) | 2026-10-21 |
| BFCL v4 `simple_python` and `multiple` | github.com/ShishirPatil/gorilla, `berkeley-function-call-leaderboard` | Apache-2.0 (stated in the BFCL README: "data used to train the models are released under Apache 2.0") | 2026-10-21 |

`data/quality/bfcl_subset.jsonl` is a seeded 200-case subset of the two BFCL categories, built by
`scripts/prepare_bfcl.py`. It is a derived copy of Apache-2.0 data. Its scores come from a
simplified checker and are not BFCL leaderboard scores.
