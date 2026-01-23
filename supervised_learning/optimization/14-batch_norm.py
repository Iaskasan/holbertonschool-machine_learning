#!/usr/bin/env python3
"""creates a batch normalization layer
for a neural network using tensorflow"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural network."""
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer
    )(prev)

    batch_norm = tf.keras.layers.BatchNormalization(
        epsilon=1e-7,
        gamma_initializer=tf.keras.initializers.Ones(),
        beta_initializer=tf.keras.initializers.Zeros()
    )(dense)

    return batch_norm
