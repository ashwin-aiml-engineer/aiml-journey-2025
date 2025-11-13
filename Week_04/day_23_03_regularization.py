"""Day 23.03 — Regularization techniques (concise)
Run time: ~12 minutes

Covers Dropout, BatchNorm (sketch), LayerNorm, weight decay (L2), and early stopping.
"""

import numpy as np

# Simple dropout implementation
def dropout(x, p=0.5, training=True):
    if not training or p == 0:
        return x
    mask = (np.random.rand(*x.shape) > p) / (1 - p)
    return x * mask

# BatchNorm (training: normalize per-batch)
def batch_norm(x, eps=1e-5):
    mu = x.mean(axis=0, keepdims=True)
    var = x.var(axis=0, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)

# Weight decay applied manually to weights (w = w - lr * weight_decay * w)

if __name__ == '__main__':
    x = np.random.randn(8, 16)
    print('Dropout (training) shape:', dropout(x, p=0.3).shape)
    print('BatchNorm mean ~0:', np.round(batch_norm(x).mean(), 3))

    # Exercises:
    # 1) Implement LayerNorm (normalize across features per sample).
    # 2) Simulate L2 weight decay during a simple SGD update loop.