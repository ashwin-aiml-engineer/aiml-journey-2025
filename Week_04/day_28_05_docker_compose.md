# Day 28.05 — Docker Compose multi-service sketch (concise)

Example services
- api: FastAPI app
- model: model server or worker
- redis: cache / queue
- postgres: metadata DB

Compose tips
- Use separate Dockerfiles and multi-stage builds for small images.
- Configure healthchecks and restart policies.
- Mount volumes for persistent DB data in dev.

Minimal sketch (copy to docker-compose.yml)

```yaml
version: '3.8'
services:
  api:
    build: ./api
    ports: ['8000:8000']
    depends_on: ['model','postgres']
  model:
    build: ./model
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - pgdata:/var/lib/postgresql/data
volumes:
  pgdata:
```

Exercise
- Create Dockerfiles for a tiny FastAPI app and run compose up locally.