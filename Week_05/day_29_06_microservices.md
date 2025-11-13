# Day 29.06 — Microservices Architecture (concise)

Service decomposition
- Split by domain or capability: api, model-serving, feature-store, worker

Communication
- Use HTTP for simple calls, gRPC for low-latency interservice calls, message broker for async

Event-driven
- Use pub-sub for decoupled workflows; implement idempotency and deduplication in consumers

Patterns
- Circuit breaker, saga for long transactions, CQRS for read/write separation

Exercise
- Draw a minimal event flow for an inference request that triggers async feature computation.