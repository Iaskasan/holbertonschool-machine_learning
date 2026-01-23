#!/usr/bin/env python3
"""sets up the Adam optimization algorithm"""
import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """sets up the Adam optimization algorithm
    Args:
        alpha (float): learning rate
        beta1 (float): momentum weight
        beta2 (float): RMSProp weight
        epsilon (float): small number to avoid division by zero
    Returns:
        tf.train.Optimizer: Adam optimizer
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=alpha,
                                         beta_1=beta1,
                                         beta_2=beta2,
                                         epsilon=epsilon)
    return optimizer
