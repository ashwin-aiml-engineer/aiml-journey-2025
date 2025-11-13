# Day 29.05 — Docker Compose Advanced (concise)

Compose overrides
- Use `docker-compose.override.yml` for local development tweaks.
- Use multiple compose files for env-specific stacks: `docker-compose -f docker-compose.yml -f docker-compose.prod.yml up`.

Healthchecks & resources
- Define `healthcheck` and `restart` policy for each service.
- Set CPU/memory limits to avoid noisy neighbors.

Networking and volumes
- Use user-defined networks for isolation; mount volumes for DB persistence.

Exercise
- Add a healthcheck to a sample service and run `docker-compose up` to observe status.