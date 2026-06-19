#!/usr/bin/env python3
"""Q-learning training."""

import numpy as np

epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Perform Q-learning."""
    total_rewards = []
    max_epsilon = epsilon

    for episode in range(episodes):
        state, _ = env.reset()
        current_reward = 0

        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, terminated, truncated, _ = env.step(action)

            if terminated and reward == 0:
                reward = -1

            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[new_state]) - Q[state, action]
            )

            current_reward += reward
            state = new_state

            if terminated or truncated:
                break

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
            -epsilon_decay * (episode + 1)
        )
        total_rewards.append(current_reward)

    return Q, total_rewards
