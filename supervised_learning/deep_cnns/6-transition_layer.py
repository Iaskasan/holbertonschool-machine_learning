#!/usr/bin/env python3
"""Builds a transition layer as described in
Densely Connected Convolutional Networks"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer for DenseNet-C.

    Args:
        X: output from the previous layer
        nb_filters: number of filters in X
        compression: compression factor (0 < θ ≤ 1)

    Returns:
        The output of the transition layer
        and the new number of filters.
    """
    he_init = K.initializers.he_normal(seed=0)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    nb_filters = int(nb_filters * compression)
    X = K.layers.Conv2D(
        filters=nb_filters,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=he_init
    )(X)
    X = K.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(X)
    return X, nb_filters
