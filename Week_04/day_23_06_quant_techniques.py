"""Day 23.06 — Quantization techniques: weight & activation helpers
Run time: ~15 minutes

- Simple helpers to quantize weights/activations, per-tensor and per-channel
- Calibration stub to pick ranges
"""

import numpy as np


def per_tensor_scale(x, num_bits=8, symmetric=True):
    qmax = 2 ** (num_bits - 1) - 1 if symmetric else 2 ** num_bits - 1
    if symmetric:
        bound = max(abs(x.min()), abs(x.max()))
        scale = bound / qmax if bound != 0 else 1.0
        return scale
    else:
        scale = (x.max() - x.min()) / qmax if x.max() != x.min() else 1.0
        return scale


def quantize_per_channel(weights, num_bits=8):
    # weights: (out_chan, in_chan, ...). Scale per out_chan
    scales = np.array([per_tensor_scale(w, num_bits=num_bits) for w in weights])
    q = np.round(weights / scales[:, None, None]).astype(np.int32)
    return q, scales

if __name__ == '__main__':
    W = np.random.randn(4, 3, 3).astype(np.float32)
    qW, scales = quantize_per_channel(W, num_bits=8)
    print('Per-channel quantized shape:', qW.shape)
    print('Scales:', np.round(scales, 5))

    # Exercises:
    # - Implement activation quantization using calibration traces (collect min/max per layer).
    # - Compare per-tensor vs per-channel MSE on a sample weight tensor.