#!/usr/bin/env python3
"""Monte Carlo policy evaluation."""

import numpy as np


def _set_state(env, state):
    """Force the environment into a specific state."""
    reset_result = env.reset()

    if hasattr(env, "state"):
        env.state = state

    if hasattr(env, "unwrapped"):
        unwrapped = env.unwrapped

        if hasattr(unwrapped, "state"):
            unwrapped.state = state

        if hasattr(unwrapped, "s"):
            if isinstance(state, tuple):
                ncols = getattr(unwrapped, "ncol", None)
                if ncols is None and hasattr(unwrapped, "desc"):
                    ncols = unwrapped.desc.shape[1]
                if ncols is not None:
                    unwrapped.s = state[0] * ncols + state[1]
            else:
                unwrapped.s = state

    return state, reset_result


def _terminal_reward(env, state):
    """Return the reward of a terminal FrozenLake-style state."""
    unwrapped = getattr(env, "unwrapped", env)
    desc = getattr(unwrapped, "desc", None)
    if desc is None:
        return None

    if isinstance(state, tuple):
        tile = desc[state]
    else:
        ncols = getattr(unwrapped, "ncol", None)
        if ncols is None and desc.ndim > 1:
            ncols = desc.shape[1]
        if ncols is None:
            tile = desc[state]
        else:
            tile = desc[state // ncols, state % ncols]

    if isinstance(tile, bytes):
        tile = tile.decode("utf-8")

    if tile == "H":
        return -1
    if tile == "G":
        return 1
    return None


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """Perform the Monte Carlo algorithm."""
    states = range(V.shape[0]) if V.ndim == 1 else np.ndindex(V.shape)

    for start_state in states:
        terminal = _terminal_reward(env, start_state)
        if terminal is not None:
            V[start_state] = terminal
            continue

        for _ in range(episodes):
            state, _ = _set_state(env, start_state)
            G = 0
            discount = 1

            for _ in range(max_steps):
                action = policy(state)
                step_result = env.step(action)

                if len(step_result) == 5:
                    new_state, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    new_state, reward, done, _ = step_result

                G += discount * reward
                state = new_state
                discount *= gamma

                if done:
                    break

            V[start_state] = V[start_state] + alpha * (G - V[start_state])

    return V
