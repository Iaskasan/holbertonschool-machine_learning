#!/usr/bin/env python3
import tensorflow as tf
"""perform a batch normalization for an entire NN before
passing to the activation function"""


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a Dense -> BatchNorm -> Activation block.

    Args:
        prev: Tensor, activated output of the previous layer
        (shape [batch, ...]).
        n: int, number of nodes (units) for the Dense layer.
        activation: callable activation function
        (e.g., tf.nn.relu, tf.nn.tanh).

    Returns:
        Tensor: activated output of the block.
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense = tf.keras.layers.Dense(
        units=n,
        use_bias=False,
        kernel_initializer=initializer
    )
    z = dense(prev)

    bn = tf.keras.layers.BatchNormalization(
        axis=-1,
        epsilon=1e-7,
        center=True,
        scale=True,
        beta_initializer='zeros',
        gamma_initializer='ones'
    )
    z_norm = bn(z)

    out = activation(z_norm)
    return out
