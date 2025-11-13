Performance metrics tracking — quick guide (10–15 minutes)

- Choose model and business metrics to track (accuracy, precision, recall, AUC, MAE, RMSE)
- Decide aggregation windows (rolling 1h, 24h, 7d)
- Track confidence intervals and sample counts for each measurement
- Implement champion/challenger comparisons and baseline tracking

Mini task:
- Given a regression model, pick 3 metrics to track and explain why (one should be MAE or RMSE).

Notes:
- For online systems, track p95 latency and error rate as human-facing KPIs as well.