#!/usr/bin/env python3
"""SGD with keras"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Creates a gradient descent optimizer with momentum in TensorFlow.

    Args:
        alpha (float): learning rate
        beta1 (float): momentum weight (0 < beta1 < 1)

    Returns:
        tf.keras.optimizers.Optimizer: momentum optimizer
    """
    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
