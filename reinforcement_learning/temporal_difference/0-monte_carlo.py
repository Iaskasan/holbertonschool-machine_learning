#!/usr/bin/env python3
"""Monte Carlo policy evaluation."""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """Perform the Monte Carlo algorithm."""
    desc = getattr(getattr(env, "unwrapped", env), "desc", None)
    ncols = None
    if desc is not None and getattr(desc, "ndim", 1) > 1:
        ncols = desc.shape[1]

    for _ in range(episodes):
        reset_result = env.reset()
        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        episode = []

        for _ in range(max_steps):
            action = policy(state)
            step_result = env.step(action)

            if len(step_result) == 5:
                new_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                new_state, reward, done, _ = step_result

            if done and reward == 0 and desc is not None and ncols is not None:
                row = new_state // ncols
                col = new_state % ncols
                if desc[row, col] == b'H':
                    reward = -1

            episode.append((state, reward))
            state = new_state

            if done:
                break

        G = 0
        visited = set()
        for state, reward in reversed(episode):
            G = reward + gamma * G
            if state in visited:
                continue
            visited.add(state)
            V[state] = V[state] + alpha * (G - V[state])

    return V
