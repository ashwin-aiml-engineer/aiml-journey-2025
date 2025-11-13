Experiment tracking — quick guide (10–15 minutes)

Why tracking matters
- Reproducibility: be able to reproduce a model result from params + code + data
- Collaboration: compare experiments, surface best runs, and avoid duplicated work
- Auditing: maintain history for compliance and post-mortems

What to log
- Parameters: hyperparameters, random seeds, data filters
- Metrics: training/validation metrics and business KPIs
- Artifacts: model files, preprocessor, plots, notebooks
- Metadata: git commit, environment, hardware, run time

Mini task
- Sketch a minimal run payload JSON you would store for each experiment.
  - Example keys: run_id, params, metrics, artifacts, git_commit, user

Notes
- Lightweight: you can log runs to local JSON/CSV for small projects; use MLflow/W&B for team scale.