#!/usr/bin/env python3
"""Load the FrozenLake environment."""

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Load a FrozenLake environment."""
    if desc is None and map_name is None:
        desc = generate_random_map(size=8)
    elif desc is not None:
        desc = ["".join(row) if isinstance(row, list) else row for row in desc]

    return gym.make(
        "FrozenLake-v1",
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery,
    )
