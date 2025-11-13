# Day 29.04 — Request Processing Optimization (concise)

Batching & async
- Batch incoming requests to improve throughput; tune batch size to latency target.
- Use async endpoints and worker queues for long-running inference.

Queue & priority
- Use priority queues for critical requests; implement backpressure when queue full.
- Offload heavy jobs to workers and return job IDs for polling.

Timeouts & throttling
- Set per-request timeouts and a global concurrency limit.
- Implement rate limiting (token bucket) at gateway or API layer.

Exercise
- Sketch a small loop that groups requests arriving within 50ms into a single batch.