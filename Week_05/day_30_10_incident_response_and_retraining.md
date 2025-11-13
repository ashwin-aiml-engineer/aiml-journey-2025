Incident response & retraining playbook (10–15 minutes)

- Detection: thresholds / anomaly detectors trigger incidents
- Triage: classify incident (data issue, model bug, infra)
- Mitigation: rollback to previous model, enable fallback, throttle traffic
- Root cause: examine feature distributions, upstream data changes, recent code deploys
- Retraining triggers: performance threshold, drift magnitude, time-based schedule
- Post-incident: post-mortem, update playbook, add regression tests

Mini task:
- Sketch a 3-step rollback plan for a model that starts returning high error rates.