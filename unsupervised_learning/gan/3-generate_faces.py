#!/usr/bin/env python3
"""Build convolutional generator and discriminator models for faces."""
from tensorflow import keras


def convolutional_GenDiscr():
    """Create and return the convolutional generator and discriminator."""

    def get_generator():
        """Build the generator model."""
        inputs = keras.Input(shape=(16,))
        x = keras.layers.Dense(4 * 4 * 128, activation="tanh")(inputs)
        x = keras.layers.Reshape((4, 4, 128))(x)
        x = keras.layers.Conv2DTranspose(
            64, kernel_size=4, strides=2, padding="same",
            activation="tanh"
        )(x)
        x = keras.layers.Conv2DTranspose(
            32, kernel_size=4, strides=2, padding="same",
            activation="tanh"
        )(x)
        outputs = keras.layers.Conv2D(
            1, kernel_size=3, padding="same", activation="tanh"
        )(x)
        return keras.Model(inputs, outputs, name="generator")

    def get_discriminator():
        """Build the discriminator model."""
        inputs = keras.Input(shape=(16, 16, 1))
        x = keras.layers.Conv2D(
            32, kernel_size=4, strides=2, padding="same",
            activation="tanh"
        )(inputs)
        x = keras.layers.Conv2D(
            64, kernel_size=4, strides=2, padding="same",
            activation="tanh"
        )(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(128, activation="tanh")(x)
        outputs = keras.layers.Dense(1, activation="tanh")(x)
        return keras.Model(inputs, outputs, name="discriminator")

    return get_generator(), get_discriminator()
