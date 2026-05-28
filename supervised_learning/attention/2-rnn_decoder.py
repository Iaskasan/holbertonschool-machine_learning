#!/usr/bin/env python3
"""RNN decoder for machine translation."""

import tensorflow as tf

SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """Decodes a target token using attention over encoder outputs."""

    def __init__(self, vocab, embedding, units, batch):
        """Set up the decoder layers."""
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
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """Decode one token step from the previous state and encoder states."""
        context, _ = self.attention(s_prev, hidden_states)
        x = tf.reshape(x, (-1, 1))
        x = self.embedding(x)
        context = tf.expand_dims(context, 1)
        x = tf.concat([context, x], axis=-1)
        output, s = self.gru(x, initial_state=s_prev)
        y = self.F(tf.squeeze(output, axis=1))
        return y, s
