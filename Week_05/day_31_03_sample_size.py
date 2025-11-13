"""Day 31.03 — Sample size for two-proportion test (approx)
Run: ~5-10 minutes

Usage: python day_31_03_sample_size.py
This uses normal approximation with common z-values for alpha=0.05 and typical power values.
"""
from __future__ import annotations

import math

Z_ALPHA_2 = {0.05: 1.96, 0.01: 2.576}
Z_POWER = {0.8: 0.84, 0.9: 1.28}


def sample_size_proportions(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Approximate per-group sample size for detecting difference between p1 and p2.
    This uses a normal approximation with common z-values for alpha and power.
    Note: For precise work, use a proper power-analysis library (statsmodels).
    """
    z_a = Z_ALPHA_2.get(alpha, 1.96)
    z_b = Z_POWER.get(power, 0.84)
    p_pool = 0.5 * (p1 + p2)
    se_pool = math.sqrt(2 * p_pool * (1 - p_pool))
    se_effect = math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    num = (z_a * se_pool + z_b * se_effect) ** 2
    denom = (p1 - p2) ** 2
    n = math.ceil(num / denom)
    return int(n)


if __name__ == "__main__":
    # quick example: baseline conversion 5% vs expected 6.5% (MDE 1.5pp)
    p1 = 0.05
    p2 = 0.065
    n = sample_size_proportions(p1, p2, alpha=0.05, power=0.8)
    print(
        f"Estimated sample size per group (p1={p1}, p2={p2}): {n}"
    )
    print(
        "Notes: uses approximate z-scores; use statsmodels for exact values."
    )
