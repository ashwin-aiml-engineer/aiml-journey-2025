"""Day 31.07 — Tiny CUPED demo (controlled experiment using pre-experiment covariate)
Run: ~10-15 minutes

This shows how to compute a simple CUPED-adjusted metric using a baseline covariate.
"""
from __future__ import annotations

import numpy as np


def cuped_adjust(y: np.ndarray, x: np.ndarray):
    """Return CUPED-adjusted y and theta (covariate coefficient).
    y: post-treatment metric (e.g., conversion)
    x: pre-experiment covariate correlated with y (e.g., historical metric)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    cov = np.cov(x, y, ddof=1)[0, 1]
    var_x = np.var(x, ddof=1)
    theta = cov / (var_x + 1e-12)
    y_adj = y - theta * (x - x.mean())
    return y_adj, float(theta)


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    # simulate pre-experiment metric x correlated with y
    x = rng.normal(0.05, 0.01, size=1000)
    noise = rng.normal(0, 0.02, size=1000)
    y = 0.1 + 0.5 * x + noise
    y_adj, theta = cuped_adjust(y, x)
    print("theta:", theta)
    print("mean y:", float(y.mean()))
    print("mean y_adj:", float(y_adj.mean()))
