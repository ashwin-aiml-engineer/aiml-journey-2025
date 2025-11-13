# Day 29.02 — API Architecture Patterns (concise)

Layering
- Repository layer: DB access and queries
- Service layer: business logic, model calls, orchestration
- API layer: request validation, response formatting

Factory & Singleton
- Use factory to load models by name/version; Singleton for shared resources (DB/clients)

Strategy & Adapter
- Strategy for model selection; Adapter for integrating different model runtimes

Testing
- Keep repository/service logic testable with mocks; use contract tests for API compatibility

Exercise
- Sketch class names and responsibilities for a small repo/service/api structure.