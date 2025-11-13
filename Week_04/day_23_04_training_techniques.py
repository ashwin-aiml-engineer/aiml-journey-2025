"""Day 23.04 — Advanced training techniques (short demos)
Run time: ~12 minutes

- Learning rate scheduler examples (step decay, cosine annealing)
- Warmup and cyclical LR sketch
"""

import math

def step_decay(initial_lr, epoch, drop=0.5, epochs_drop=10):
    return initial_lr * (drop ** math.floor((1 + epoch) / epochs_drop))

def cosine_annealing(initial_lr, epoch, T_max):
    return initial_lr * (1 + math.cos(math.pi * epoch / T_max)) / 2

if __name__ == '__main__':
    for e in [0, 5, 10, 15]:
        print('step decay lr at epoch', e, '->', round(step_decay(0.01, e), 6))
    for e in [0, 10, 20]:
        print('cosine annealing lr at epoch', e, '->', round(cosine_annealing(0.01, e, 20), 6))

    # Exercises:
    # - Implement a linear warmup (lr = base * epoch / warmup_epochs for early epochs).
    # - Implement a cyclical LR (triangular) schedule and plot it.