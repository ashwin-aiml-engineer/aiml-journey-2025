"""Day 23.09 — Quantization trade-offs (quick scaffold)
Run time: ~12 minutes

- Emulate an accuracy vs size experiment using a toy classifier and simulated quantization
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


def simulate_size(bits, base_params=100000):
    # rough bytes estimate: params * bits / 8
    return base_params * bits / 8

if __name__ == '__main__':
    X = np.random.randn(500, 16)
    y = (X[:, 0] + 0.1 * np.random.randn(500) > 0).astype(int)
    clf = LogisticRegression(max_iter=200).fit(X, y)
    acc = clf.score(X, y)
    print('FP32 (sim) accuracy:', round(acc, 4), 'size (bytes):', simulate_size(32))
    for bits in [16, 8, 4]:
        # naive assumption: accuracy degrades linearly with log2(bits)
        simulated_acc = acc - (0.02 * (32 - bits) / 8)
        print(f'{bits}-bit sim acc:', round(simulated_acc, 4), 'size:', simulate_size(bits))

    # Exercises:
    # - Replace naive accuracy model with actual quantized retrain on a small NN.
    # - Measure inference time vs model size (use time.perf_counter).