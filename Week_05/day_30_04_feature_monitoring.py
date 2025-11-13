"""Day 30.04 — Feature monitoring checks (missing rate, cardinality, outliers)
Run: ~10 minutes

Small utilities to compute missing value rate, cardinality, and detect large cardinality changes.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Tuple


def missing_rate(arr: Iterable) -> float:
    arr = list(arr)
    n = len(arr)
    if n == 0:
        return 0.0
    missing = sum(1 for v in arr if v is None)
    return missing / n


def cardinality(arr: Iterable) -> int:
    return len(set(arr))


def top_k_categories(arr: Iterable, k: int = 5) -> List[Tuple[object, int]]:
    c = Counter(arr)
    return c.most_common(k)


if __name__ == "__main__":
    sample = ["a", "b", None, "a", "c", "b", None, "d"]
    print("missing_rate:", missing_rate(sample))
    print("cardinality:", cardinality(sample))
    print("top_k:", top_k_categories(sample, k=3))
