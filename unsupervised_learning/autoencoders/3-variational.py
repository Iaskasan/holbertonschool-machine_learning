#!/usr/bin/env python3
"""Variational autoencoder module."""

import tensorflow.keras as keras
from tensorflow.keras import backend as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.

    Args:
        input_dims: Integer, dimensions of the model input.
        hidden_layers: List of nodes for encoder hidden layers.
        latent_dims: Integer, dimensions of latent space.

    Returns:
        encoder: Encoder model.
        decoder: Decoder model.
        auto: Full variational autoencoder model.
    """

    def sampling(args):
        """Reparameterization trick."""
        mean, log_var = args
        epsilon = K.random_normal(shape=K.shape(mean))
        return mean + K.exp(log_var / 2) * epsilon

    # Encoder
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input

    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation="relu")(x)

    mean = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    latent = keras.layers.Lambda(sampling)([mean, log_var])

    encoder = keras.Model(
        encoder_input,
        [latent, mean, log_var],
        name="encoder"
    )

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

    # Full VAE
    auto_input = keras.Input(shape=(input_dims,))
    z, z_mean, z_log_var = encoder(auto_input)
    auto_output = decoder(z)

    auto = keras.Model(auto_input, auto_output, name="autoencoder")

    # KL divergence loss
    kl_loss = -0.5 * K.sum(
        1 + z_log_var - K.square(z_mean) - K.exp(z_log_var),
        axis=1
    )

    auto.add_loss(K.mean(kl_loss))

    auto.compile(
        optimizer="adam",
        loss="binary_crossentropy"
    )

    return encoder, decoder, auto
