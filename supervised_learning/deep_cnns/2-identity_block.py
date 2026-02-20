#!/usr/bin/env python3
"""Builds an identity block as described in
Deep Residual Learning for Image Recognition (2015)
"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block
    """
    F11, F3, F12 = filters
    he_init = K.initializers.he_normal(seed=0)
    X = K.layers.Conv2D(filters=F11, kernel_size=(1, 1),
                        padding='valid', kernel_initializer=he_init)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(filters=F3, kernel_size=(3, 3),
                        padding='same', kernel_initializer=he_init)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(filters=F12, kernel_size=(1, 1),
                        padding='valid', kernel_initializer=he_init)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Add()([X, A_prev])
    X = K.layers.Activation('relu')(X)
    return X
