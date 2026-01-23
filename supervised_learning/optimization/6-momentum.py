#!/usr/bin/env python3
"""sets up the gradient descent
with momentum optimization algorithm
"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """sets up the gradient descent
    with momentum optimization algorithm"""
    optimizer = tf.keras.optimizers.SGD(alpha, beta1)
    return optimizer
