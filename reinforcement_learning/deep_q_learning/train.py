#!/usr/bin/env python3
"""Train a DQN agent to play Atari Breakout with keras-rl2."""

from pathlib import Path

import gymnasium as gym
import keras
import tensorflow.keras as tf_keras
from gymnasium.wrappers import AtariPreprocessing
from keras import Sequential
from keras.layers import Conv2D, Dense, Flatten, Permute
from tensorflow.keras.optimizers.legacy import Adam

if not hasattr(tf_keras, "__version__"):
    tf_keras.__version__ = keras.__version__

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy


ENV_NAME = "ALE/Breakout-v5"
INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4
NB_STEPS = 1000000
POLICY_PATH = Path(__file__).resolve().with_name("policy.h5")


class GymnasiumCompatibilityWrapper(gym.Wrapper):
    """Expose the legacy Gym API expected by keras-rl2."""

    def reset(self, **kwargs):
        observation, _ = self.env.reset(**kwargs)
        return observation

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return observation, reward, done, info

    def render(self, mode="human"):
        return self.env.render()


class AutoFireWrapper(gym.Wrapper):
    """Automatically relaunch Breakout when the game is waiting for FIRE."""

    def __init__(self, env):
        super().__init__(env)
        self.fire_action = self._get_fire_action()
        self.lives = 0

    def _get_fire_action(self):
        try:
            action_meanings = self.env.unwrapped.get_action_meanings()
        except AttributeError:
            return None

        for action_name in ("FIRE", "RIGHTFIRE", "LEFTFIRE"):
            if action_name in action_meanings:
                return action_meanings.index(action_name)
        return None

    def _get_lives(self):
        try:
            return self.env.unwrapped.ale.lives()
        except AttributeError:
            return 0

    def _press_fire(self):
        if self.fire_action is None:
            return None
        return self.env.step(self.fire_action)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.lives = self._get_lives()

        fire_step = self._press_fire()
        if fire_step is not None:
            observation, _, terminated, truncated, info = fire_step
            if terminated or truncated:
                observation, info = self.env.reset(**kwargs)
                self.lives = self._get_lives()
            else:
                self.lives = self._get_lives()

        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        lives = self._get_lives()

        if 0 < lives < self.lives:
            fire_step = self._press_fire()
            if fire_step is not None:
                (
                    observation,
                    fire_reward,
                    fire_terminated,
                    fire_truncated,
                    info,
                ) = fire_step
                reward += fire_reward
                terminated = terminated or fire_terminated
                truncated = truncated or fire_truncated
                lives = self._get_lives()

        self.lives = lives
        return observation, reward, terminated, truncated, info


def build_env(render_mode=None):
    """Create a Breakout environment compatible with keras-rl2."""
    env = gym.make(ENV_NAME, render_mode=render_mode, frameskip=1)
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=INPUT_SHAPE[0],
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False,
    )
    env = AutoFireWrapper(env)
    return GymnasiumCompatibilityWrapper(env)


def build_model(nb_actions):
    """Create the convolutional Q-network used by the DQN agent."""
    model = Sequential()
    model.add(
        Permute((2, 3, 1), input_shape=(WINDOW_LENGTH,) + INPUT_SHAPE)
    )
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation="relu"))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation="relu"))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(nb_actions, activation="linear"))
    return model


def build_agent(model, nb_actions):
    """Create a DQN agent configured for Atari control."""
    memory = SequentialMemory(limit=1_000_000, window_length=WINDOW_LENGTH)
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.1,
        value_test=0.05,
        nb_steps=500000,
    )
    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        nb_steps_warmup=50000,
        target_model_update=5000,
        policy=policy,
        gamma=0.99,
        train_interval=4,
        delta_clip=1.0,
    )
    return dqn


def main():
    """Train the agent and save the learned policy network."""
    env = build_env()
    nb_actions = env.action_space.n

    model = build_model(nb_actions)
    dqn = build_agent(model, nb_actions)
    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])
    dqn.fit(env, nb_steps=NB_STEPS, visualize=False, verbose=2)
    dqn.model.save(str(POLICY_PATH))
    env.close()


if __name__ == "__main__":
    main()
