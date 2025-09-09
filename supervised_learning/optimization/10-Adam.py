#!/usr/bin/env python3
"""Adam optimiser via keras"""
import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
    Creates an Adam optimizer in TensorFlow.

    Args:
        alpha (float): learning rate
        beta1 (float): weight for first moment
        beta2 (float): weight for second moment
        epsilon (float): small constant to avoid division by zero

    Returns:
        tf.keras.optimizers.Adam: configured Adam optimizer
    """
    return tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon
    )
