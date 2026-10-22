# Planner worked example

`apg plan` turns measured capacity into a deployment size. This example is also a test
(`test_plan_matches_hand_calculation` in `tests/test_report_plan.py`), so the two can't drift.

Inputs (synthetic, for the test only, not a measurement):

- An engine whose worst-of-repeats capacity at the `agent-step` SLO is 16 concurrent sessions.
- 5 sessions completed in the 10 s window at that level, so 1,800 sessions/hour, and
  200 output tokens/s.
- 200 users, 1 concurrent agent session each, so a demand of 200 sessions.
- A full A40 priced at 0.90 USD/hour (a test value). One slice is the whole GPU (`slice_fraction` 1.0), so 0.90 USD/hour.

By hand:

- Slices needed = ceil(200 / 16) = 13.
- Utilisation = 200 / (13 × 16) = 96%.
- Cost per hour = 13 × 0.90 = 11.70 USD.
- Cost per task at capacity = 0.90 / 1,800 = 0.0005 USD.
- Cost per 1M output tokens at capacity = 0.90 / (200 × 3600) × 1e6 = 1.25 USD.

```bash
uv run apg plan --users 200 --slo agent-step --a40-usd-per-hour 0.9
```

`apg plan` prints the same numbers and lists its assumptions. Costs show `n/a` until a real GPU
price is set in `configs/matrix.yaml` (`cost.a40_usd_per_hour`) with its source and date.
