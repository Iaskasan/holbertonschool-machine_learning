#!/usr/bin/env python3
"""Training helper for the transformer translation model."""

import tensorflow as tf

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Transformer learning rate schedule with warmup."""

    def __init__(self, dm, warmup_steps=4000):
        """Initialize the schedule."""
        super().__init__()
        self.dm = tf.cast(dm, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """Return the learning rate for a training step."""
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred):
    """Compute masked sparse categorical crossentropy loss."""
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction='none'
    )
    loss = loss_object(real, pred)
    mask = tf.cast(tf.math.not_equal(real, 0), loss.dtype)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def accuracy_function(real, pred):
    """Compute masked token accuracy."""
    predictions = tf.argmax(pred, axis=2, output_type=real.dtype)
    matches = tf.cast(tf.equal(real, predictions), tf.float32)
    mask = tf.cast(tf.math.not_equal(real, 0), tf.float32)
    matches *= mask
    return tf.reduce_sum(matches) / tf.reduce_sum(mask)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """Create and train a transformer for Portuguese-to-English MT."""
    dataset = Dataset(batch_size, max_len)
    input_vocab = dataset.tokenizer_pt.vocab_size + 2
    target_vocab = dataset.tokenizer_en.vocab_size + 2

    transformer = Transformer(
        N, dm, h, hidden, input_vocab, target_vocab, max_len, max_len
    )
    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    @tf.function
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        encoder_mask, combined_mask, decoder_mask = create_masks(
            inp, tar_inp
        )

        with tf.GradientTape() as tape:
            predictions = transformer(
                inp,
                tar_inp,
                training=True,
                encoder_mask=encoder_mask,
                look_ahead_mask=combined_mask,
                decoder_mask=decoder_mask
            )
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, transformer.trainable_variables)
        )

        accuracy = accuracy_function(tar_real, predictions)
        train_loss.update_state(loss)
        train_accuracy.update_state(accuracy)

    for epoch in range(epochs):
        train_loss.reset_state()
        train_accuracy.reset_state()

        for batch, (inp, tar) in enumerate(dataset.data_train):
            train_step(inp, tar)

            if (batch + 1) % 50 == 0:
                print(
                    "Epoch {}, batch {}: loss {} accuracy {}".format(
                        epoch + 1,
                        batch + 1,
                        train_loss.result().numpy(),
                        train_accuracy.result().numpy()
                    )
                )

        print(
            "Epoch {}: loss {} accuracy {}".format(
                epoch + 1,
                train_loss.result().numpy(),
                train_accuracy.result().numpy()
            )
        )

    return transformer
