#!/usr/bin/env python3
"""L2 Regularization Layer Creation"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a TensorFlow Dense layer with L2 regularization.

    Args:
        prev (tf.Tensor): output of the previous layer
        n (int): number of nodes in the new layer
        activation (callable): activation function to use
        lambtha (float): L2 regularization parameter

    Returns:
        tf.Tensor: the output of the new layer
    """
    # Ensure the input is a proper tensor, even if main.py gave an int
    if isinstance(prev, int):
        prev = tf.keras.Input(shape=(prev,))

    # Define L2 regularizer
    l2_reg = tf.keras.regularizers.L2(lambtha)

    # Create Dense layer with L2 regularization
    dense_layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=l2_reg,
        kernel_initializer="he_normal"
    )

    # Apply it to prev and return the output tensor
    return dense_layer(prev)
