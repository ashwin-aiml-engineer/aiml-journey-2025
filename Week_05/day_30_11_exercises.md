Day 30 — Exercises (pick 1–3, 10–20 minutes each)

1) Data drift quick check
- Run `day_30_02_data_drift_methods.py` on your dataset (split historical vs recent) and report PSI and KS.

2) Stream detector
- Use `day_30_03_concept_drift_methods.py` to simulate a drift and observe detection points.

3) Feature check
- Run `day_30_04_feature_monitoring.py` on a sample table and report missing rates and top categories.

4) Prediction monitoring
- Run `day_30_05_prediction_monitoring.py`, compute confidence fractions, and pick thresholds.

5) Dashboard sketch
- Use `day_30_09_dashboards_and_viz.py` to plot a metric over time and annotate drift points.

Capstone mini-project (optional, 1–2 hours)
- Build a tiny monitoring pipeline using the above components:
  - Periodically compute PSI and KS between a reference window and current window.
  - Log metrics to a CSV or simple timeseries store.
  - Trigger a local "alert" (print) when PSI > 0.2 or KS > 0.1, and record the event.

Notes
- These exercises are intentionally small and runnable without heavy infra. Replace local prints with real DB/alert integrations as needed.