#!/usr/bin/env python3
"""Play an episode using a trained Q-table."""


def play(env, Q, max_steps=100):
    """Play one episode by always exploiting the Q-table."""
    state, _ = env.reset()
    total_rewards = 0
    renders = []

    render = env.render()
    renders.append(render)
    print(render, end="")

    for _ in range(max_steps):
        action = Q[state].argmax()
        state, reward, terminated, truncated, _ = env.step(action)
        total_rewards += reward

        render = env.render()
        renders.append(render)
        print(render, end="")

        if terminated or truncated:
            break

    return total_rewards, renders
