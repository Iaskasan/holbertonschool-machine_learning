#!/usr/bin/env python3
"""RMSProp with keras"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Creates an RMSProp optimizer in TensorFlow.

    Args:
        alpha (float): learning rate
        beta2 (float): RMSProp weight (discounting factor, usually 0.9)
        epsilon (float): small constant to avoid division by zero

    Returns:
        tf.keras.optimizers.RMSprop: configured optimizer
    """
    return tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
