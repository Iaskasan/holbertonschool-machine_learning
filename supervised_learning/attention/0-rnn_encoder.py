#!/usr/bin/env python3
"""RNN encoder for machine translation."""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """Encodes an input sequence using an embedding layer and a GRU."""

    def __init__(self, vocab, embedding, units, batch):
        """Set up the encoder layers."""
        super().__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

    def initialize_hidden_state(self):
        """Return an initial zero hidden state."""
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """Run the encoder on an input token sequence."""
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
