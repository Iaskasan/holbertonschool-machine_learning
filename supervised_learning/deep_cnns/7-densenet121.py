#!/usr/bin/env python3
"""Builds the DenseNet-121 architecture as described in
Densely Connected Convolutional Networks (Huang et al., 2017)"""
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Builds the DenseNet-121 architecture.

    Args:
        growth_rate: growth rate of the network
        compression: compression factor for transition layers

    Returns:
        The Keras DenseNet-121 model.
    """
    he_init = K.initializers.he_normal(seed=0)
    input_layer = K.Input(shape=(224, 224, 3))

    X = K.layers.BatchNormalization(axis=3)(input_layer)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(
        64, (7, 7), strides=(2, 2), padding='same',
        kernel_initializer=he_init
    )(X)
    X = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(X)
    nb_filters = 64
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 6)
    X, nb_filters = transition_layer(X, nb_filters, compression)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)
    X, nb_filters = transition_layer(X, nb_filters, compression)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)
    X, nb_filters = transition_layer(X, nb_filters, compression)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.AveragePooling2D(pool_size=(7, 7), padding='same')(X)

    X = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer=he_init)(X)

    model = K.models.Model(inputs=input_layer, outputs=X)
    return model
