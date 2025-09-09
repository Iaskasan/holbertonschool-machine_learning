#!/usr/bin/env python3
"""perform a batch normalization to an entire NN
before passing to the activation function"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a Dense -> BatchNorm -> Activation block.

    Args:
        prev: Tensor, activated output of the previous
        layer (shape [batch, ...]).
        n: int, number of nodes (units) for the Dense layer.
        activation: activation to apply; can be a callable (e.g., tf.nn.relu)
                    or a string (e.g., "relu", "tanh", "sigmoid").

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

    act = tf.keras.activations.get(activation)
    out = act(z_norm)

    return out
