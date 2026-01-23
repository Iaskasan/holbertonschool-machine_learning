#!/usr/bin/env python3
"""creates a learning rate decay operation"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """creates a learning rate decay operation
    Args:
        alpha (float): initial learning rate
        decay_rate (float): weight used to determine the rate at which
                           the learning rate decays
        decay_step (int): number of passes of gradient descent that should
                          occur before updating the learning rate
    Returns:
        tf.Tensor: learning rate decay operation
    """
    lr_decay = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
    return lr_decay
