A/B Testing fundamentals — quick checklist (10–15 minutes)

Core ideas
- Define hypothesis: what change do you expect? (e.g., model B increases conversion by X%)
- Choose primary metric (business-aligned) and 1–2 guardrail metrics
- Decide sample size and experiment duration (power, MDE)
- Randomization: user-id hashing for sticky assignment
- Significance: choose alpha (commonly 0.05) and power (0.8 or 0.9)

Mini task (15 min)
- Write a hypothesis for your model change and list the primary/secondary metrics.
  - Example: "Model B increases conversion rate by 1.5 percentage points".

Notes
- Keep experiments as simple as possible: one primary metric, pre-registered analysis plan.
- Log assignment metadata (seed, model version, experiment id) with each prediction.