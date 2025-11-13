Model versioning fundamentals — quick checklist (10–15 minutes)

Core ideas
- Semantic versioning for models: MAJOR.MINOR.PATCH (useful for compatibility notes)
- Lifecycle stages: development → staging → production → archived
- Track model metadata: version id, training commit hash, data snapshot id, hyperparameters
- Backward compatibility: input schema, output contract, and constraints
- Migration strategies: shadowing/champion-challenger, blue-green, canary

Mini task
- For a deployed model, list the metadata fields you'd store when registering a version. Example:
  - model_name, version, git_commit, train_data_hash, training_date, metrics

Notes
- Use a model registry (MLflow Model Registry or similar) when you have multiple models and teams.