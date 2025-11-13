A/B testing infra checklist (10 minutes)

Minimal components
- Feature flag / assignment service (hash-based, sticky assignments)
- Logging and metadata: experiment id, variant, timestamp, user id, model version
- Experiment config store and rollout rules (percentages, cohorts)
- Metrics pipeline: aggregated metrics to timeseries DB; online and batch aggregation
- Monitoring: dashboards for primary metric, guardrails and alerts for anomalies

Mini task
- List the fields to log for each request (user_id, exp_id, variant, metric, ts, model_version).