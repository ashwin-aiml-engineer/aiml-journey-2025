"""Day 32.04 — Tiny JSON-based model registry stub
Run: ~10 minutes

This is a tiny local registry to register and lookup models by name and version.
It writes a small JSON file `model_registry.json` in the current directory.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

REG_PATH = "model_registry.json"


def _load():
    if not os.path.exists(REG_PATH):
        return {}
    with open(REG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(data):
    with open(REG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def register(model_name: str, version: str, metadata: dict):
    data = _load()
    model_entry = data.setdefault(model_name, {})
    model_entry[version] = {
        "metadata": metadata,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    _save(data)
    print(f"Registered {model_name}:{version}")


def get_versions(model_name: str):
    data = _load()
    return list(data.get(model_name, {}).keys())


def get_metadata(model_name: str, version: str):
    data = _load()
    return data.get(model_name, {}).get(version)


if __name__ == "__main__":
    # demo registration
    register(
        "demo_model",
        "1.0.0",
        {"git_commit": "abc123", "metrics": {"val_loss": 0.42}},
    )
    print("versions:", get_versions("demo_model"))
    print("meta:", get_metadata("demo_model", "1.0.0"))
