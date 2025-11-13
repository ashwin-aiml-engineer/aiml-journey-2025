Day 31 — Exercises (pick 1–3, 10–20 minutes each)

1) Sample size quick-check
- Run `day_31_03_sample_size.py` for a realistic MDE and record the per-group sample size.

2) Traffic split validation
- Run `day_31_04_traffic_splitting.py` for a sample user set and verify sticky assignment across runs.

3) Bandit comparison
- Run `day_31_05_bandits.py` and compare total rewards between epsilon-greedy and Thompson Sampling.

4) Stat test practice
- Use `day_31_06_stat_tests.py` to compute t-stat and z-stat on synthetic samples.

5) CUPED demo
- Run `day_31_07_cuped.py` and observe variance reduction in the adjusted metric.

Capstone mini-project (optional, 1–2 hours)
- Implement a small experiment runner that:
  - Assigns users via hash-based splitter
  - Simulates a binary outcome using two different rates
  - Aggregates results and computes proportions z-test and CUPED-adjusted difference
  - Logs results to CSV and prints decision (promote/don't promote)

Notes
- These examples are intentionally small and dependency-light. Replace prints with real logging/DB in production.
