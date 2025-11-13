"""Day 31.04 — Traffic splitting demo: hash-based assignment and sticky vs non-sticky
Run: ~5-10 minutes
"""
from __future__ import annotations

import hashlib


def assign_user(user_id: str, buckets: int = 100) -> int:
    """Deterministic hash-based assignment returning a bucket in [0, buckets-1]."""
    h = hashlib.md5(user_id.encode("utf-8")).hexdigest()
    return int(h, 16) % buckets


def assign_variant(user_id: str, variants=("control", "treatment")) -> str:
    b = assign_user(user_id, buckets=100)
    # simple 50/50 split by bucket
    idx = 0 if b < 50 else 1
    return variants[idx]


if __name__ == "__main__":
    users = [f"user_{i}" for i in range(10)]
    for u in users:
        print(u, assign_variant(u))
