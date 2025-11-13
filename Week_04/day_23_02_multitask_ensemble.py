"""Day 23.02 — Multi-task learning + small ensemble pattern
Run time: ~12 minutes

- Shows how to structure multi-task heads from a shared trunk
- Demonstrates a tiny ensemble (averaging) using scikit-learn's simple estimators
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Shared trunk (feature extractor) - simple projection
def trunk(x):
    W = np.random.randn(x.shape[1], 16) * 0.1
    return np.tanh(x.dot(W))

# Multi-task heads: classification + regression heads (toy)
def multi_task_heads(features):
    # classification logits
    clf_logits = features.dot(np.random.randn(16, 2) * 0.1)
    # regression output
    reg_out = features.dot(np.random.randn(16, 1) * 0.1).squeeze(-1)
    return clf_logits, reg_out

# Tiny ensemble: average predictions of different estimators
if __name__ == '__main__':
    X = np.random.randn(200, 8)
    y = np.random.randint(0, 2, 200)

    # Train two simple estimators and average probs
    clf1 = LogisticRegression(max_iter=200).fit(X, y)
    clf2 = DecisionTreeClassifier(max_depth=5).fit(X, y)

    p1 = clf1.predict_proba(X)
    p2 = clf2.predict_proba(X)
    p_avg = (p1 + p2) / 2
    preds = p_avg.argmax(axis=1)
    print('Ensemble accuracy (toy):', (preds == y).mean())

    # Multi-task demo
    feat = trunk(X)
    logits, reg = multi_task_heads(feat)
    print('Multi-task: logits shape', logits.shape, 'regression shape', reg.shape)

    # Exercises:
    # - Add an auxiliary loss from an intermediate layer and show how to weight it.
    # - Replace averaging ensemble with a small meta-learner trained on predictions.