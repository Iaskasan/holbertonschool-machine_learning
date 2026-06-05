#!/usr/bin/env python3
"""Dataset loader with TensorFlow encoding helpers."""

import tensorflow as tf
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """Load the translation dataset and prepare tokenizers."""

    def __init__(self):
        """Load, tokenize, and wrap dataset splits."""
        self.data_train = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            split="train",
            as_supervised=True,
            try_gcs=True
        )
        self.data_valid = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            split="validation",
            as_supervised=True,
            try_gcs=True
        )
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """Create Portuguese and English sub-word tokenizers."""
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased",
            use_fast=True
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            use_fast=True
        )

        def iterator(lang_index, batch_size=1024):
            for batch in data.batch(batch_size):
                texts = batch[lang_index].numpy()
                yield [text.decode("utf-8") for text in texts]

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            iterator(0),
            vocab_size=2 ** 13
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            iterator(1),
            vocab_size=2 ** 13
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Encode a Portuguese/English translation pair."""
        pt_tokens = self.tokenizer_pt(
            pt.numpy().decode("utf-8"),
            add_special_tokens=False
        )["input_ids"]
        en_tokens = self.tokenizer_en(
            en.numpy().decode("utf-8"),
            add_special_tokens=False
        )["input_ids"]

        pt_tokens = [self.tokenizer_pt.vocab_size] + pt_tokens + [
            self.tokenizer_pt.vocab_size + 1
        ]
        en_tokens = [self.tokenizer_en.vocab_size] + en_tokens + [
            self.tokenizer_en.vocab_size + 1
        ]

        np = transformers.utils.generic.np
        return np.array(pt_tokens), np.array(en_tokens)

    def tf_encode(self, pt, en):
        """TensorFlow wrapper around encode."""
        pt_tokens, en_tokens = tf.py_function(
            self.encode,
            [pt, en],
            [tf.int64, tf.int64]
        )
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])
        return pt_tokens, en_tokens
