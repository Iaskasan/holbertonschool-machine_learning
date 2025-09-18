#!/usr/bin/env python3
"""Create a layer with L2 regularization in TensorFlow"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a Dense layer with L2 regularization.

    Args:
        prev: tensor, output of the previous layer
        n: int, number of nodes in the new layer
        activation: activation function to use
        lambtha: float, L2 regularization parameter

    Returns:
        The output tensor of the new layer
    """
    l2_regularizer = tf.keras.regularizers.L2(lambtha)

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=l2_regularizer,
        kernel_initializer="he_normal"
    )

    return layer(prev)
