"""Day 23.01 — FFN deep dive, Dense layers, Skip/Residual demo
Run time: ~10 minutes

- Shows a simple feedforward network built with numpy
- Demonstrates a skip connection (residual) pattern
- Short exercises at the bottom
"""

import numpy as np

# Simple dense layer (no frameworks) for didactic purposes
class Dense:
    def __init__(self, in_dim, out_dim, activation=None):
        self.W = np.random.randn(in_dim, out_dim) * 0.1
        self.b = np.zeros(out_dim)
        self.activation = activation

    def __call__(self, x):
        z = x.dot(self.W) + self.b
        if self.activation == 'relu':
            return np.maximum(0, z)
        return z

# Build a small FFN
def ffn(x):
    h1 = Dense(4, 8, activation='relu')(x)
    h2 = Dense(8, 8, activation='relu')(h1)
    out = Dense(8, 2)(h2)
    return out

# Residual block: input added to layer output (requires matching dims)
def residual_block(x):
    h = Dense(4, 4, activation='relu')(x)
    h = Dense(4, 4)(h)
    return x + h  # skip connection

if __name__ == '__main__':
    x = np.random.randn(5, 4)
    print('FFN output shape:', ffn(x).shape)
    print('Residual block output shape (should match input):', residual_block(x).shape)

    # Exercises (try these):
    # 1) Add BatchNorm-like rescaling after Dense (simple mean/std normalization).
    # 2) Change network depth vs width and observe output shape and param count.
    # 3) Implement a simple skip connection that uses a 1x1 projection (Dense) when dims differ.