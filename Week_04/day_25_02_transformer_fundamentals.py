"""Day 25.02 — Transformer fundamentals: attention toy demo
Run time: ~10-15 minutes

- Small, self-contained numpy demo computing scaled dot-product attention
- Helps build intuition for queries, keys, values and softmax attention
"""

import numpy as np


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V):
    # Q,K,V: (seq_len, d)
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)
    attn = softmax(scores, axis=-1)
    return attn @ V, attn

if __name__ == '__main__':
    seq_len = 4
    d = 8
    Q = np.random.randn(seq_len, d)
    K = np.random.randn(seq_len, d)
    V = np.random.randn(seq_len, d)
    out, attn = scaled_dot_product_attention(Q, K, V)
    print('Output shape:', out.shape)
    print('Attention matrix shape:', attn.shape)

    # Exercises:
    # - Visualize attention weights (attn) as a heatmap for a toy input.
    # - Implement multi-head attention by splitting dims and concatenating outputs.