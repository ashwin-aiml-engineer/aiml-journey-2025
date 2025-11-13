"""Day 26.09 — Training config: LR schedules, warmup, grad accumulation sketch
Run time: ~12 minutes

- Small helpers and pseudocode for warmup, cosine decay, and gradient accumulation
"""

import math


def linear_warmup(base_lr, step, warmup_steps):
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)
    return base_lr


def cosine_decay(base_lr, step, total_steps):
    return base_lr * 0.5 * (1 + math.cos(math.pi * step / total_steps))


def apply_grad_accum(optimizer_step_fn, accumulation_steps):
    # pseudocode: call optimizer_step_fn every accumulation_steps
    def wrapper(step, grads):
        # accumulate grads externally and call optimizer_step_fn when needed
        pass
    return wrapper

if __name__ == '__main__':
    for s in [0, 5, 10]:
        print('warmup lr at step', s, '->', round(linear_warmup(1e-4, s, 10), 7))
    for s in [0, 50, 100]:
        print('cosine lr at step', s, '->', round(cosine_decay(1e-4, s, 100), 7))

    # Exercises:
    # - Implement small loop that prints LR for 100 steps using warmup + cosine decay.
    # - Sketch how to accumulate grads for batch_size=256 using accumulation_steps.