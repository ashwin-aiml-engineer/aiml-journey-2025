Day 32 — Exercises (pick 1–3, 10–20 minutes each)

1) MLflow quickstart
- Run `day_32_03_mlflow_quickstart.py`. If mlflow isn't installed, inspect the printed steps and try locally after `pip install mlflow`.

2) Local registry
- Run `day_32_04_local_registry.py` to register a demo model and inspect `model_registry.json`.

3) Experiment logging
- Run `day_32_05_experiment_logging.py` and open `runs.jsonl` to view recorded runs.

Capstone mini-project (optional, 1–2 hours)
- Build a mini registry+tracker:
  - Use the `local_registry` stub and `experiment_logging` stub to register a few model versions and log runs
  - Compare runs for different versions and pick the best to promote

Notes
- These examples are intentionally small and dependency-light. For team-scale, integrate MLflow, DVC, and hosted experiment platforms.
