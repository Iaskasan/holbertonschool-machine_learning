#!/usr/bin/env python3
"""Convert a gensim Word2Vec model to a Keras Embedding layer."""

import tensorflow as tf


def gensim_to_keras(model):
    """
    Convert a trained gensim Word2Vec model to a trainable Keras Embedding.

    Args:
        model: trained gensim Word2Vec model

    Returns:
        A trainable keras Embedding layer initialized with gensim weights
    """
    weights = model.wv.vectors

    embedding = tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        trainable=True
    )
    embedding.build((None,))
    embedding.set_weights([weights])

    return embedding
