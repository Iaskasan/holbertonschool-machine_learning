#!/usr/bin/env python3
"""Builds a dense block as described in
Densely Connected Convolutional Networks"""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block using the bottleneck DenseNet-B structure.

    Args:
        X: output from the previous layer
        nb_filters: number of filters in X
        growth_rate: growth rate for the dense block
        layers: number of layers in the dense block

    Returns:
        The concatenated output of the dense block,
        and the number of filters within the concatenated outputs.
    """
    he_init = K.initializers.he_normal(seed=0)

    for i in range(layers):
        bn1 = K.layers.BatchNormalization(axis=3)(X)
        relu1 = K.layers.Activation('relu')(bn1)
        conv1 = K.layers.Conv2D(
            4 * growth_rate,
            kernel_size=(1, 1),
            padding='same',
            kernel_initializer=he_init
        )(relu1)

        bn2 = K.layers.BatchNormalization(axis=3)(conv1)
        relu2 = K.layers.Activation('relu')(bn2)
        conv2 = K.layers.Conv2D(
            growth_rate,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=he_init
        )(relu2)

        X = K.layers.Concatenate(axis=3)([X, conv2])

        nb_filters += growth_rate

    return X, nb_filters
