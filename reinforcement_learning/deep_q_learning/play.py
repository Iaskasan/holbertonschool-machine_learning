#!/usr/bin/env python3
"""Load a trained DQN policy and record Atari Breakout playback."""

from collections import Counter
from pathlib import Path

import cv2
import gymnasium as gym
import keras
import tensorflow.keras as tf_keras
from gymnasium.wrappers import AtariPreprocessing
from keras.models import load_model
from PIL import Image
from tensorflow.keras.optimizers.legacy import Adam

if not hasattr(tf_keras, "__version__"):
    tf_keras.__version__ = keras.__version__

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy


ENV_NAME = "ALE/Breakout-v5"
INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4
POLICY_PATH = Path(__file__).resolve().with_name("policy.h5")
VIDEO_PATH = Path(__file__).resolve().with_name("playback.mp4")
GIF_PATH = Path(__file__).resolve().with_name("playback.gif")
DISPLAY_SCALE = 4
FPS = 30
MAX_STEPS = 2000


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


def build_env():
    """Create a Breakout environment that returns frames for recording."""
    env = gym.make(ENV_NAME, render_mode="rgb_array", frameskip=1)
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


def scale_frame(frame):
    """Upscale a recorded Atari frame for easier viewing."""
    height, width = frame.shape[:2]
    return cv2.resize(
        frame,
        (width * DISPLAY_SCALE, height * DISPLAY_SCALE),
        interpolation=cv2.INTER_NEAREST,
    )


def create_video_writer(frame):
    """Create an OpenCV writer for the playback video."""
    scaled_frame = scale_frame(frame)
    height, width = scaled_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(VIDEO_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (width, height),
    )
    return writer, scaled_frame


def save_gif(frames):
    """Fallback to an animated GIF when MP4 writing is unavailable."""
    images = [Image.fromarray(scale_frame(frame)) for frame in frames]
    images[0].save(
        GIF_PATH,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / FPS),
        loop=0,
    )


def get_action_meanings(env):
    """Return human-readable action names when the backend exposes them."""
    try:
        return env.unwrapped.get_action_meanings()
    except AttributeError:
        return [str(index) for index in range(env.action_space.n)]


def record_episode(agent, env):
    """Play one greedy episode and save it to disk."""
    agent.training = False
    agent.reset_states()
    observation = env.reset()
    episode_reward = 0.0
    done = False
    step = 0
    frames = []
    writer = None
    action_counts = Counter()
    action_meanings = get_action_meanings(env)

    frame = env.render()
    if frame is not None:
        frames.append(frame)
        writer, scaled_frame = create_video_writer(frame)
        if writer.isOpened():
            writer.write(cv2.cvtColor(scaled_frame, cv2.COLOR_RGB2BGR))
        else:
            writer.release()
            writer = None

    while not done and step < MAX_STEPS:
        action = agent.forward(observation)
        action_counts[int(action)] += 1
        observation, reward, done, _ = env.step(action)
        agent.backward(reward, terminal=done)
        episode_reward += reward
        step += 1
        frame = env.render()
        if frame is not None:
            frames.append(frame)
            if writer is not None:
                scaled_frame = scale_frame(frame)
                writer.write(cv2.cvtColor(scaled_frame, cv2.COLOR_RGB2BGR))

    action_summary = {
        action_meanings[index]: count
        for index, count in sorted(action_counts.items())
    }

    if writer is not None:
        writer.release()
        return episode_reward, VIDEO_PATH, step, done, action_summary

    save_gif(frames)
    return episode_reward, GIF_PATH, step, done, action_summary


def main():
    """Load the saved policy network and record one played episode."""
    env = build_env()
    model = load_model(str(POLICY_PATH), compile=False)
    memory = SequentialMemory(limit=1, window_length=WINDOW_LENGTH)
    policy = GreedyQPolicy()

    dqn = DQNAgent(
        model=model,
        nb_actions=env.action_space.n,
        memory=memory,
        nb_steps_warmup=0,
        target_model_update=1,
        policy=policy,
    )
    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    try:
        episode_reward, output_path, steps, done, action_summary = record_episode(
            dqn, env
        )
        print(f"Episode reward: {episode_reward}")
        print(f"Steps recorded: {steps}")
        print(f"Episode finished naturally: {done}")
        print(f"Action counts: {action_summary}")
        print(f"Saved playback to: {output_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
