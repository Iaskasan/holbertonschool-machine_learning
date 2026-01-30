#!/usr/bin/env python3
"""
creates a neural network layer in
tensorFlow that includes L2 regularization
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a dense layer with L2 regularization

    Args:
        prev: tensor, output of the previous layer
        n: int, number of nodes in the layer to create
        activation: activation function for the layer
        lambtha: float, L2 regularization parameter

    Returns:
        tensor, output of the layer
    """
    regularizer = tf.keras.regularizers.L2(lambtha)
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=regularizer,
        kernel_initializer="he_normal"
    )
    return layer(prev)
