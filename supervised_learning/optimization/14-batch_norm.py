#!/usr/bin/env python3
"""creates a batch normalization layer
for a neural network using tensorflow"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer with BN applied BEFORE activation.

    Args:
        prev (tf.Tensor): Output of previous layer (can be already activated).
        n (int): Number of nodes in the new layer.
        activation (callable or str): Activation function to apply.

    Returns:
        tf.Tensor: Activated output of the batch-normalized layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense = tf.keras.layers.Dense(
        units=n,
        activation=None,
        kernel_initializer=initializer
    )(prev)

    batch_norm = tf.keras.layers.BatchNormalization(
        epsilon=1e-7,
        gamma_initializer=tf.keras.initializers.Ones(),
        beta_initializer=tf.keras.initializers.Zeros()
    )(dense)

    if activation is not None:
        output = tf.keras.layers.Activation(activation)(batch_norm)
    else:
        output = batch_norm

    return output
