# Day 28.03 — Model Registry & Management (concise)

Core ideas
- Store artifacts with metadata: model id, version, training data, metrics, hyperparams
- Use a registry (MLflow, DVC, or custom) to track versions and re-use models
- Promote models: staging -> production, and enable rollback

Best practices
- Save evaluation metrics and dataset hashes alongside model artifacts.
- Automate model validation and canary release for new versions.

Exercise
- Draft a minimal model card template with fields: id, version, author, dataset, metrics, notes.