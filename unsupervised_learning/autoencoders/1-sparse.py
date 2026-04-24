#!/usr/bin/env python3
"""Sparse autoencoder module."""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Creates a sparse autoencoder.

    Args:
        input_dims: Integer, dimensions of the model input.
        hidden_layers: List of integers, number of nodes for each
            hidden layer in the encoder.
        latent_dims: Integer, dimensions of the latent space.
        lambtha: L1 regularization parameter on the encoded output.

    Returns:
        encoder: The encoder model.
        decoder: The decoder model.
        auto: The full sparse autoencoder model.
    """
    # Encoder
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input

    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation="relu")(x)

    latent = keras.layers.Dense(
        latent_dims,
        activation="relu",
        activity_regularizer=keras.regularizers.l1(lambtha)
    )(x)

    encoder = keras.Model(encoder_input, latent, name="encoder")

    # Decoder
    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input

    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation="relu")(x)

    decoder_output = keras.layers.Dense(
        input_dims,
        activation="sigmoid"
    )(x)

    decoder = keras.Model(decoder_input, decoder_output, name="decoder")

    # Full autoencoder
    auto_input = keras.Input(shape=(input_dims,))
    encoded = encoder(auto_input)
    decoded = decoder(encoded)

    auto = keras.Model(auto_input, decoded, name="autoencoder")

    auto.compile(
        optimizer="adam",
        loss="binary_crossentropy"
    )

    return encoder, decoder, auto
