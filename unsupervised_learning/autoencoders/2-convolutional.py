#!/usr/bin/env python3
"""Convolutional autoencoder module."""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder.

    Args:
        input_dims: Tuple containing input dimensions.
        filters: List containing filters for encoder conv layers.
        latent_dims: Tuple containing latent space dimensions.

    Returns:
        encoder: Encoder model.
        decoder: Decoder model.
        auto: Full autoencoder model.
    """
    # Encoder
    encoder_input = keras.Input(shape=input_dims)
    x = encoder_input

    for f in filters:
        x = keras.layers.Conv2D(
            f, (3, 3), padding="same", activation="relu"
        )(x)
        x = keras.layers.MaxPooling2D((2, 2), padding="same")(x)

    encoder = keras.Model(encoder_input, x, name="encoder")

    # Decoder
    decoder_input = keras.Input(shape=latent_dims)
    x = decoder_input

    for f in reversed(filters[1:]):
        x = keras.layers.Conv2D(
            f, (3, 3), padding="same", activation="relu"
        )(x)
        x = keras.layers.UpSampling2D((2, 2))(x)

    x = keras.layers.Conv2D(
        filters[0], (3, 3), padding="valid", activation="relu"
    )(x)
    x = keras.layers.UpSampling2D((2, 2))(x)

    decoder_output = keras.layers.Conv2D(
        input_dims[-1], (3, 3), padding="same", activation="sigmoid"
    )(x)

    decoder = keras.Model(decoder_input, decoder_output, name="decoder")

    # Full autoencoder
    auto_input = keras.Input(shape=input_dims)
    encoded = encoder(auto_input)
    decoded = decoder(encoded)

    auto = keras.Model(auto_input, decoded, name="autoencoder")

    auto.compile(
        optimizer="adam",
        loss="binary_crossentropy"
    )

    return encoder, decoder, auto
