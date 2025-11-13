"""Day 30.09 — Small dashboard / viz helpers (matplotlib fallback)
Run: ~10 minutes

Creates a simple histogram and timeseries plot for metrics.
"""
from __future__ import annotations

try:
    import matplotlib.pyplot as plt
    has_mpl = True
except ImportError:
    has_mpl = False

import numpy as np


def plot_hist(data, title="hist"):
    if not has_mpl:
        print("matplotlib not installed — describe: show histogram of data")
        return
    plt.figure()
    plt.hist(data, bins=30)
    plt.title(title)
    plt.show()


def plot_series(x, y, title="series"):
    if not has_mpl:
        print("matplotlib not installed — describe: show timeseries")
        return
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    data = rng.normal(size=1000)
    plot_hist(data, title="Prediction distribution")
    x = np.arange(100)
    y = np.cumsum(rng.normal(scale=0.1, size=100))
    plot_series(x, y, title="Rolling metric (example)")
