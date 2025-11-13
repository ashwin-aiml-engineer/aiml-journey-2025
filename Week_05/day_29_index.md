# Day 29: Advanced Model Serving with FastAPI & Docker Compose

Short practice files (each ~15 minutes) covering advanced FastAPI patterns, multi-model serving, request optimization, caching, Docker Compose advanced, microservices, monitoring and CI/CD.

Files:
- `day_29_01_fastapi_patterns.md` — dependency injection, background tasks, WebSocket, middleware
- `day_29_02_api_architecture.md` — repository/service/factory patterns and design notes
- `day_29_03_multi_model_serving.py` — runnable FastAPI stub for multi-model routing (safe fallback)
- `day_29_04_request_optimization.md` — batching, async handling, queues, throttling
- `day_29_05_docker_compose_advanced.md` — compose overrides, healthchecks, resource limits
- `day_29_06_microservices.md` — service decomposition, message brokers, event-driven patterns
- `day_29_07_caching_redis.py` — Redis cache-aside stub (safe fallback)
- `day_29_08_monitoring_logging.md` — Prometheus, tracing, structured logs checklist
- `day_29_09_ci_cd_deploy.md` — CI/CD sketch for build/test/deploy with Actions
- `day_29_10_exercises.md` — 12 focused 15-min exercises

How to use
- Run Python stubs from repo root, e.g.:
  ```powershell
  python Week_05\day_29_03_multi_model_serving.py
  python Week_05\day_29_07_caching_redis.py
  ```
- Read md notes for quick concepts and follow exercises for hands-on practice.