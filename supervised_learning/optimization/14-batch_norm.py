#!/usr/bin/env python3
"""Batch normalisation of a layer"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a Dense -> BatchNorm -> Activation block.

    Args:
        prev: Tensor, activated output of the previous layer
        (shape [batch_size, ...]).
        n: int, number of units for the Dense layer.
        activation: callable or string, activation to apply
        (e.g., tf.nn.relu or "relu").

    Returns:
        Tensor: the activated output of the block.
    """
    # 1) Linear layer (no bias: beta will play that role after normalization)
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense = tf.keras.layers.Dense(
        units=n,
        use_bias=False,
        kernel_initializer=initializer
    )
    z = dense(prev)
    batch_mean, batch_var = tf.nn.moments(z, axes=[0])

    gam = tf.Variable(tf.ones([n]), trainable=True, name=f"{dense.name}_gamma")
    bet = tf.Variable(tf.zeros([n]), trainable=True, name=f"{dense.name}_beta")

    # 4) Normalize + scale/shift (epsilon avoids division by zero)
    eps = 1e-7
    z_hat = tf.nn.batch_normalization(
        x=z,
        mean=batch_mean,
        variance=batch_var,
        offset=bet,     # beta (shift)
        scale=gam,     # gamma (scale)
        variance_epsilon=eps
    )  # shape: [batch_size, n]

    # 5) Activation (accepts callable or string)
    if activation is None:
        return z_hat
    act_fn = tf.keras.activations.get(activation)
    return act_fn(z_hat)
