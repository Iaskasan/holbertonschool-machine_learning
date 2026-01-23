#!/usr/bin/env python3
"""creates a batch normalization layer
for a neural network using tensorflow"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """creates a batch normalization layer
    for a neural network using tensorflow

    prev: the activated output of the previous layer
    n: number of nodes in the layer to be created
    activation: activation function that the layer should use

    Returns: the activated output of the batch normalization layer
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    gamma = tf.keras.initializers.Ones()
    beta = tf.keras.initializers.Zeros()
    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=initializer,
    )(prev)

    batch_norm = tf.keras.layers.BatchNormalization(
        axis=-1,
        gamma_initializer=gamma,
        beta_initializer=beta
    )(dense)

    if activation is not None:
        activated_output = activation(batch_norm)
        return activated_output

    return batch_norm
