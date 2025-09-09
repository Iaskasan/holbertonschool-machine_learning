#!/usr/bin/env python3
"""time-based decay of alpha with keras"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a time-based (inverse time) learning rate schedule in TensorFlow.

    Args:
        alpha (float): initial learning rate
        decay_rate (float): decay factor
        decay_step (int): number of steps before applying decay

    Returns:
        tf.keras.optimizers.schedules.InverseTimeDecay: learning rate schedule
    """
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
