"""Day 31.06 — Small stat-test helpers: Welch t-test and proportions z-test
Run: ~10 minutes
"""
from __future__ import annotations

import math
import numpy as np

try:
    from scipy import stats  # optional, used if available for p-values
    has_scipy = True
except ImportError:
    has_scipy = False


def welch_ttest(a: np.ndarray, b: np.ndarray):
    """Return t-statistic and degrees of freedom for Welch's t-test.
    If scipy is available, also return the two-sided p-value.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    na, nb = len(a), len(b)
    ma, mb = a.mean(), b.mean()
    sa, sb = a.var(ddof=1), b.var(ddof=1)
    t = (ma - mb) / math.sqrt(sa / na + sb / nb)
    df = (sa / na + sb / nb) ** 2 / ((sa ** 2) / (na ** 2 * (na - 1)) + (sb ** 2) / (nb ** 2 * (nb - 1)))
    if has_scipy:
        p = stats.t.sf(abs(t), df) * 2
        return float(t), float(df), float(p)
    return float(t), float(df), None


def proportions_z_test(p1: float, n1: int, p2: float, n2: int):
    """Compute z-statistic and p-value for difference in proportions (approx).
    If scipy available, compute p-value from normal CDF.
    """
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / (se + 1e-12)
    if has_scipy:
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        return float(z), float(p)
    # rough p-value using normal approx constants (alpha guidance)
    return float(z), None


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, size=100)
    b = rng.normal(0.2, 1, size=100)
    print("Welch t-test:", welch_ttest(a, b))
    print(
        "Proportions z-test (example):",
        proportions_z_test(0.05, 1000, 0.06, 1000),
    )
