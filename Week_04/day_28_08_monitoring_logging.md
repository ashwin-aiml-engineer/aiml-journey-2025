# Day 28.08 — Monitoring & Logging (concise)

Logging
- Use structured logs (JSON) with timestamps and request IDs.
- Centralize logs using ELK, Loki, or cloud logging services.

Metrics
- Export metrics (Prometheus) for latency, error rates, throughput.
- Monitor model metrics: accuracy drift, input distribution changes.

Alerts & tracing
- Add alert rules for latency/error spikes.
- Use distributed tracing (Jaeger) for request flow analysis.

Exercise
- Add a log line format example and a Prometheus metric name for inference latency.