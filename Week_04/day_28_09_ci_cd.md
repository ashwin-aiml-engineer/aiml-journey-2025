# Day 28.09 — CI/CD Quick Checklist (concise)

Key steps
- Run unit tests and lint on commits (GitHub Actions).
- Build Docker images and run smoke tests in CI.
- Push images to registry and deploy to staging with a workflow.
- Run post-deploy checks and integration tests.

Example (GitHub Actions job sketch)
```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: pip install -r requirements.txt
      - run: pytest -q
```

Exercise
- Write a short GitHub Actions workflow that builds and runs tests.