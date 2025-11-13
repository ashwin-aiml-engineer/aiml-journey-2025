"""Day 23.07 — Model compression: low-rank (SVD) and pruning sketch
Run time: ~15 minutes

- Demonstrates SVD-based low-rank approximation for weight matrices
- Shows a simple magnitude-based pruning example
"""

import numpy as np


def low_rank_approx(W, rank):
    # W: (in_dim, out_dim)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    S[rank:] = 0
    return (U * S) @ Vt


def magnitude_prune(W, prune_ratio=0.5):
    flat = np.abs(W).ravel()
    thresh = np.percentile(flat, prune_ratio * 100)
    W_pruned = W * (np.abs(W) >= thresh)
    return W_pruned

if __name__ == '__main__':
    W = np.random.randn(64, 64)
    W_lr = low_rank_approx(W, rank=8)
    print('Low-rank approx shape:', W_lr.shape)
    W_pr = magnitude_prune(W, prune_ratio=0.7)
    print('Pruned nonzeros:', (W_pr != 0).sum())

    # Exercises:
    # - Measure reconstruction MSE between W and its low-rank approx for varying ranks.
    # - Implement iterative pruning + fine-tuning loop (sketch).