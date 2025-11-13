# Day 28.01 — Complete ML Application Architecture (concise)

Key components
- Ingress/API layer (FastAPI), model inference services, data services, auth service
- Async workers for heavy tasks (Celery/RQ) and message queue (RabbitMQ/Redis)
- Storage: object store for raw files, DB for metadata, feature store for features
- Monitoring: logs, metrics, tracing

Design tips
- Keep services stateless where possible for horizontal scaling.
- Use async or worker queues for long-running work.
- Design clear API contracts and version them (v1, v2).

Security
- Use HTTPS, API keys or OAuth, and input validation.
- Limit public exposure of model endpoints; use internal networks for heavy models.