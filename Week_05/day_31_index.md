Day 31 — A/B Testing for ML Models

Run time: ~10–20 minutes per exercise (pick 1–2 for a short session)

Overview

This day covers experimental design and online testing for ML models. Files here are concise 15-minute practice items: brief theory + small runnable demos or checklists.

Files created:
- `day_31_01_fundamentals.md` — A/B testing fundamentals and KPI choices
- `day_31_02_experimental_design.md` — design options: between/within, stratified, sequential
- `day_31_03_sample_size.py` — quick sample-size calculator for proportions (approx)
- `day_31_04_traffic_splitting.py` — hash-based assignment and sticky vs non-sticky demo
- `day_31_05_bandits.py` — epsilon-greedy and Thompson Sampling Bernoulli bandit sims
- `day_31_06_stat_tests.py` — t-test (Welch) and proportions z-test (with scipy fallback for p-values)
- `day_31_07_cuped.py` — small CUPED variance-reduction demo
- `day_31_08_infrastructure.md` — infra checklist: feature flags, experiment platform, logging
- `day_31_09_variance_reduction.md` — short notes on CUPED, stratification, regression adj.
- `day_31_10_exercises.md` — pickable exercises and a mini capstone

Quick run examples (PowerShell):

```powershell
python .\Week_05\day_31_03_sample_size.py
python .\Week_05\day_31_04_traffic_splitting.py
python .\Week_05\day_31_05_bandits.py
python .\Week_05\day_31_06_stat_tests.py
```

Choose one short demo and run it to see outputs and example usage.