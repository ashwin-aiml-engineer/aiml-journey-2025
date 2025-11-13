"""Day 23.05 — Quantization fundamentals (simulation)
Run time: ~10 minutes

- Demonstrates FP32 -> simulated INT8 quantization via symmetric linear mapping
- Shows how precision reduction changes dynamic range
"""

import numpy as np


def fake_quantize_tensor(x, num_bits=8, symmetric=True):
    # simple uniform quantization to signed integers in [-Q, Q]
    qmax = 2 ** (num_bits - 1) - 1 if symmetric else 2 ** num_bits - 1
    min_x, max_x = x.min(), x.max()
    if symmetric:
        bound = max(abs(min_x), abs(max_x))
        scale = bound / qmax if bound != 0 else 1.0
        q = np.round(x / scale).astype(np.int32)
        q = np.clip(q, -qmax, qmax)
        return (q * scale).astype(x.dtype)
    else:
        scale = (max_x - min_x) / qmax if max_x != min_x else 1.0
        q = np.round((x - min_x) / scale).astype(np.int32)
        q = np.clip(q, 0, qmax)
        return (q * scale + min_x).astype(x.dtype)

if __name__ == '__main__':
    x = np.linspace(-3, 3, 20).astype(np.float32)
    print('Original:', x[:6], '...')
    q = fake_quantize_tensor(x, num_bits=8, symmetric=True)
    print('Quantized (8-bit symmetric):', q[:6], '...')
    q4 = fake_quantize_tensor(x, num_bits=4, symmetric=True)
    print('Quantized (4-bit symmetric):', q4[:6], '...')

    # Exercises:
    # - Compare MSE between original and quantized tensors for different bit widths.
    # - Try asymmetric quantization (symmetric=False) and observe differences.