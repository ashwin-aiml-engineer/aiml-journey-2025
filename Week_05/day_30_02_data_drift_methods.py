"""Day 30.02 — Data drift detection: PSI, KS, Jensen-Shannon (quick demos)
Run: ~10-15 minutes

This file provides small, dependency-light implementations of:
- Population Stability Index (PSI)
- Kolmogorov-Smirnov statistic (KS)
- Jensen-Shannon divergence (JS)

Each function prints a small example when invoked as __main__.
"""
from __future__ import annotations

import math
import numpy as np


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Compute Population Stability Index between two numeric arrays.
    Lower PSI ≈ no change, higher indicates drift. This is a bucketed PSI.
    """
    eps = 1e-8
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    # create quantile-based bins from expected
    cuts = np.quantile(expected, np.linspace(0, 1, buckets + 1))
    psi_val = 0.0
    for i in range(buckets):
        lo, hi = cuts[i], cuts[i + 1]
        exp_cnt = ((expected >= lo) & (expected <= hi)).sum()
        act_cnt = ((actual >= lo) & (actual <= hi)).sum()
        exp_pct = exp_cnt / max(1, len(expected))
        act_pct = act_cnt / max(1, len(actual))
        exp_pct = max(exp_pct, eps)
        act_pct = max(act_pct, eps)
        psi_val += (exp_pct - act_pct) * math.log(exp_pct / act_pct)
    return float(psi_val)


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Kolmogorov-Smirnov statistic (D) between two samples.
    Returns the maximum absolute difference between empirical CDFs.
    """
    a = np.sort(np.asarray(a))
    b = np.sort(np.asarray(b))
    data = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, data, side="right") / len(a)
    cdf_b = np.searchsorted(b, data, side="right") / len(b)
    d = np.max(np.abs(cdf_a - cdf_b))
    return float(d)


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence for discrete distributions.
    Accepts probability vectors or raw counts (will normalize).
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    # normalize
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)
    
    def kl(x, y):
        mask = (x > 0)
        return np.sum(
            x[mask] * np.log(x[mask] / (y[mask] + 1e-12))
        )
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


if __name__ == "__main__":
    # quick demo data
    rng = np.random.default_rng(42)
    ref = rng.normal(loc=0.0, scale=1.0, size=1000)
    # create a shifted sample to simulate drift
    cur = rng.normal(loc=0.3, scale=1.2, size=1000)

    print("PSI:", psi(ref, cur, buckets=10))
    print("KS statistic:", ks_statistic(ref, cur))

    # discrete distribution example for JS
    p = np.array([10, 20, 30, 40])
    q = np.array([12, 18, 25, 45])
    print("Jensen-Shannon:", jensen_shannon(p, q))
