#!/usr/bin/env python3
"""Initialize the Q-table."""

import numpy as np


def q_init(env):
    """Return a zero-initialized Q-table for the environment."""
    return np.zeros((env.observation_space.n, env.action_space.n))
