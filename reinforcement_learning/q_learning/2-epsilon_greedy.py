#!/usr/bin/env python3
"""Epsilon-greedy action selection."""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Determine the next action using epsilon-greedy."""
    p = np.random.uniform(0, 1)

    if p < epsilon:
        return np.random.randint(Q.shape[1])
    return np.argmax(Q[state])
