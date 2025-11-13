# Day 28.04 — API Development Best Practices (concise)

Quick checklist
- Use FastAPI for typed endpoints and auto OpenAPI docs.
- Version your API (e.g., /api/v1/predict).
- Validate inputs with Pydantic models and sanitize text/file uploads.
- Use consistent response format: {"status":200,"result":...}

Security & performance
- Protect endpoints with JWT or API keys.
- Add rate limiting and CORS configuration.
- Use caching for repeated predictions and batch requests for throughput.

Exercise
- Sketch a FastAPI POST /predict endpoint with a Pydantic input model.