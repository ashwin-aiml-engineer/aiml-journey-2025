"""Day 32.05 — Simple file-based experiment logger (params + metrics)
Run: ~5-10 minutes

Logs each run to a JSONL file `runs.jsonl` with a run_id, params and metrics.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

RUNS_PATH = "runs.jsonl"


def log_run(params: dict, metrics: dict, artifacts: list | None = None):
    run = {
        "run_id": str(uuid.uuid4()),
        "params": params,
        "metrics": metrics,
        "artifacts": artifacts or [],
        "git_commit": params.get("git_commit"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(RUNS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(run) + "\n")
    print("Logged run:", run["run_id"])
    return run["run_id"]


if __name__ == "__main__":
    run_id = log_run({"lr": 0.01, "git_commit": "abc123"}, {"val_loss": 0.42})
    print("Done. run_id:", run_id)
