#!/usr/bin/env python3
"""Dataset loader and tokenizer preparation for translation."""

import tensorflow_datasets as tfds
import transformers


class _LazyTokenizer:
    """Tokenizer proxy that loads the real tokenizer on first use."""

    def __init__(self, dataset, lang_index, model_name):
        """Store the data source and pretrained model name."""
        self._dataset = dataset
        self._lang_index = lang_index
        self._model_name = model_name
        self._tokenizer = None

    def _iterator(self, batch_size=1024):
        """Yield decoded text batches for tokenizer training."""
        for batch in self._dataset.batch(batch_size):
            texts = batch[self._lang_index].numpy()
            yield [text.decode("utf-8") for text in texts]

    def _load(self):
        """Instantiate and train the tokenizer once."""
        if self._tokenizer is None:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self._model_name,
                use_fast=True
            )
            self._tokenizer = tokenizer.train_new_from_iterator(
                self._iterator(),
                vocab_size=2 ** 13
            )
        return self._tokenizer

    def __call__(self, *args, **kwargs):
        """Proxy direct tokenizer calls."""
        return self._load()(*args, **kwargs)

    def __getattr__(self, name):
        """Proxy attribute access to the underlying tokenizer."""
        return getattr(self._load(), name)


class Dataset:
    """Load the translation dataset and prepare tokenizers."""

    def __init__(self):
        """Load dataset splits and set up tokenizer attributes."""
        self.data_train, self.data_valid = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            split=["train", "validation"],
            as_supervised=True
        )
        self.tokenizer_pt = _LazyTokenizer(
            self.data_train,
            0,
            "neuralmind/bert-base-portuguese-cased"
        )
        self.tokenizer_en = _LazyTokenizer(
            self.data_train,
            1,
            "bert-base-uncased"
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

        self.tokenizer_pt = tokenizer_pt
        self.tokenizer_en = tokenizer_en
        return tokenizer_pt, tokenizer_en
