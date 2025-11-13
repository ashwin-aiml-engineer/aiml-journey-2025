"""Day 24.02 — Convolution & pooling demo (numpy)
Run time: ~15 minutes

- Implements a 2D convolution and max pooling in numpy for intuition
"""

import numpy as np


def conv2d(image, kernel, stride=1, padding=0):
    # image: HxW, kernel: kh x kw
    if padding:
        image = np.pad(image, padding, mode='constant')
    H, W = image.shape
    kh, kw = kernel.shape
    out_h = (H - kh) // stride + 1
    out_w = (W - kw) // stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch = image[i*stride:i*stride+kh, j*stride:j*stride+kw]
            out[i, j] = np.sum(patch * kernel)
    return out


def max_pool(x, pool=2, stride=2):
    H, W = x.shape
    out_h = (H - pool) // stride + 1
    out_w = (W - pool) // stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch = x[i*stride:i*stride+pool, j*stride:j*stride+pool]
            out[i, j] = patch.max()
    return out

if __name__ == '__main__':
    img = np.linspace(0, 1, 64).reshape(8, 8)
    kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    c = conv2d(img, kernel, stride=1, padding=1)
    p = max_pool(c, pool=2, stride=2)
    print('conv shape:', c.shape, 'pool shape:', p.shape)

    # Exercises:
    # - Modify stride and padding and observe output shapes.
    # - Implement a simple multi-channel conv by summing per-channel results.