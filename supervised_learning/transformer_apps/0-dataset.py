#!/usr/bin/env python3
"""Dataset loader and tokenizer preparation for translation."""

import tensorflow_datasets as tfds
import transformers


class Dataset:
    """Load the translation dataset and prepare tokenizers."""

    def __init__(self):
        """Load dataset splits and build tokenizers fromm the train split."""
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
