"""Day 30.03 — Concept drift stream detectors (Page-Hinkley, simple ADWIN-like)
Run: ~10-15 minutes

Includes:
- Page-Hinkley test (simple implementation for streaming scalar values)
- A simple adaptive-window detector (very small demo, not production-ready)
"""
from __future__ import annotations

import math


class PageHinkley:
    """Simple Page-Hinkley implementation for a numeric stream.
    Usage: create detector, call update(x) for each value. Use check() to
    see whether drift is detected.
    """

    def __init__(
        self,
        delta: float = 0.005,
        lambda_: float = 50,
        alpha: float = 1 - 0.0001,
    ):
        self.delta = delta
        self.lambda_ = lambda_
        self.alpha = alpha
        self.mean = 0.0
        self.cumulative = 0.0
        self.n = 0

    def update(self, val: float) -> bool:
        self.n += 1
        prev_mean = self.mean
        self.mean = prev_mean + (val - prev_mean) / self.n
        self.cumulative = self.cumulative + val - self.mean - self.delta
        if self.cumulative > self.lambda_:
            # reset and report drift
            self._reset()
            return True
        return False

    def _reset(self) -> None:
        self.mean = 0.0
        self.cumulative = 0.0
        self.n = 0


class SimpleAdwin:
    """Tiny adaptive window simulator: keeps a short and long window and
    compares their means. This is NOT the real ADWIN implementation but
    an educational approximation.
    """

    def __init__(
        self,
        short_w: int = 50,
        long_w: int = 200,
        threshold: float = 0.1,
    ):
        self.short_w = short_w
        self.long_w = long_w
        self.threshold = threshold
        self.short_buffer: list[float] = []
        self.long_buffer: list[float] = []

    def update(self, val: float) -> bool:
        self.short_buffer.append(val)
        self.long_buffer.append(val)
        if len(self.short_buffer) > self.short_w:
            self.short_buffer.pop(0)
        if len(self.long_buffer) > self.long_w:
            self.long_buffer.pop(0)
        if len(self.long_buffer) < self.long_w:
            return False
        short_mean = sum(self.short_buffer) / len(self.short_buffer)
        long_mean = sum(self.long_buffer) / len(self.long_buffer)
        return abs(short_mean - long_mean) > self.threshold


if __name__ == "__main__":
    import random

    ph = PageHinkley(lambda_=30)
    ad = SimpleAdwin()

    # simulate a stable stream then a drift
    for i in range(300):
        value = (
            random.gauss(0, 1)
            if i < 180
            else random.gauss(0.8, 1.1)
        )
        if ph.update(value):
            print(f"Page-Hinkley detected drift at index {i}")
        if ad.update(value):
            print(f"ADWIN-like detected drift at index {i}")
