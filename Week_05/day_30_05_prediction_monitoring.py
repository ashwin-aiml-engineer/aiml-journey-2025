"""Day 30.05 — Prediction monitoring: distribution and confidence checks
Run: ~10 minutes

Small demo to compute prediction distribution statistics and confidence score summaries.
"""
from __future__ import annotations

import numpy as np


def prediction_distribution_stats(preds: np.ndarray):
    preds = np.asarray(preds)
    return {
        "min": float(preds.min()),
        "max": float(preds.max()),
        "mean": float(preds.mean()),
        "std": float(preds.std()),
        "p25": float(np.percentile(preds, 25)),
        "p50": float(np.percentile(preds, 50)),
        "p75": float(np.percentile(preds, 75)),
    }


def confidence_summary(probs: np.ndarray):
    probs = np.asarray(probs)
    # fraction above common thresholds
    return {
        ">0.9": float((probs > 0.9).mean()),
        ">0.8": float((probs > 0.8).mean()),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    preds = rng.normal(0.1, 0.5, size=1000)
    probs = rng.random(1000)
    print("pred stats:", prediction_distribution_stats(preds))
    print("confidence:", confidence_summary(probs))
