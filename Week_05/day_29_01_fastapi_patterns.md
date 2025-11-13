# Day 29.01 — Advanced FastAPI Patterns (concise)

Dependency Injection
- Use `Depends` to inject DB, auth, or model loader instances into endpoints.

Background tasks
- Use `BackgroundTasks` for non-blocking post-response work (logging, async writes).

WebSocket & SSE
- Use `WebSocket` for real-time bi-directional streams; SSE for server -> client streaming.

Middleware & hooks
- Implement middleware for request-id, auth checks, and timing.
- Use startup/shutdown events to load and warm models.

Response models
- Use Pydantic models for consistent, versioned responses.

Quick tip
- Keep endpoint handlers thin; delegate logic to service layer for testability.