#!/usr/bin/env python3
"""creates a layer of a neural network using dropout"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network with dropout.

    Args:
        prev: tensor, output of previous layer
        n: int, number of nodes in the new layer
        activation: activation function
        keep_prob: probability of keeping a neuron
        training: boolean, whether in training mode

    Returns:
        tensor, output of the new layer
    """
    # 1️⃣ Linear + activation
    layer = tf.keras.layers.Dense(units=n, activation=activation)(prev)

    # 2️⃣ Apply manual dropout only during training
    if training and keep_prob < 1.0:
        # Generate a random mask with same shape as layer output
        mask = tf.random.uniform(shape=tf.shape(layer)) < keep_prob
        mask = tf.cast(mask, dtype=layer.dtype)

        # Apply mask and scale (inverted dropout)
        layer = layer * mask / keep_prob

    return layer
