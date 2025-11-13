"""Day 31.05 — Bandit sims: epsilon-greedy and Thompson Sampling for Bernoulli arms
Run: ~10-15 minutes
"""
from __future__ import annotations

import numpy as np


def epsilon_greedy(true_rates, eps=0.1, rounds=1000, seed=0):
    rng = np.random.default_rng(seed)
    n_arms = len(true_rates)
    counts = np.zeros(n_arms, dtype=int)
    values = np.zeros(n_arms, dtype=float)
    rewards = []
    for _ in range(rounds):
        if rng.random() < eps:
            arm = rng.integers(0, n_arms)
        else:
            arm = int(np.argmax(values))
        reward = rng.random() < true_rates[arm]
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]
        rewards.append(reward)
    return counts, values, sum(rewards)


def thompson_sampling(true_rates, rounds=1000, seed=0):
    rng = np.random.default_rng(seed)
    n_arms = len(true_rates)
    successes = np.zeros(n_arms, dtype=int)
    failures = np.zeros(n_arms, dtype=int)
    rewards = []
    for _ in range(rounds):
        samples = [rng.beta(1 + successes[i], 1 + failures[i]) for i in range(n_arms)]
        arm = int(np.argmax(samples))
        reward = rng.random() < true_rates[arm]
        if reward:
            successes[arm] += 1
        else:
            failures[arm] += 1
        rewards.append(reward)
    return successes + failures, successes / (successes + failures + 1e-12), sum(rewards)


if __name__ == "__main__":
    rates = [0.05, 0.06, 0.08]
    c, v, r = epsilon_greedy(rates, eps=0.1, rounds=2000)
    print("Epsilon-greedy: counts:", c)
    print("Epsilon-greedy: est rates:", v)
    print("Epsilon-greedy: total rewards:", r)
    n, est, r2 = thompson_sampling(rates, rounds=2000)
    print("Thompson: pulls:", n)
    print("Thompson: est:", np.round(est, 4))
    print("Thompson: total rewards:", r2)
