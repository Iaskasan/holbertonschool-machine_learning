#!/usr/bin/env python3
"""sets up the RMSProp optimization algorithm using tensorflow"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """sets up the RMSProp optimization algorithm
    Args:
        alpha (float): learning rate
        beta2 (float): RMSProp weight
        epsilon (float): small number to avoid division by zero
    Returns:
        tf.train.Optimizer: RMSProp optimizer
    """
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=alpha,
                                            rho=beta2,
                                            epsilon=epsilon)
    return optimizer
