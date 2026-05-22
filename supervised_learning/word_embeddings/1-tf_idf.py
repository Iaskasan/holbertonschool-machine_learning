#!/usr/bin/env python3
"""TF-IDF embedding module."""

import re
import numpy as np


def tokenize(sentence):
    """Normalize and tokenize a sentence."""
    sentence = sentence.lower()

    # Remove possessive 's
    sentence = re.sub(r"'s\b", "", sentence)

    # Remove punctuation
    sentence = re.sub(r"[^\w\s]", "", sentence)

    return sentence.split()


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix.

    Args:
        sentences: list of sentences to analyze
        vocab: list of vocabulary words to use

    Returns:
        embeddings, features
    """
    tokenized = [tokenize(sentence) for sentence in sentences]

    if vocab is None:
        features = sorted(set(
            word
            for sentence in tokenized
            for word in sentence
        ))
    else:
        features = vocab

    features = np.array(features)

    s = len(sentences)
    f = len(features)

    embeddings = np.zeros((s, f))

    # Compute document frequencies
    df = np.zeros(f)

    for j, word in enumerate(features):
        for sentence in tokenized:
            if word in sentence:
                df[j] += 1

    # Compute IDF
    idf = np.zeros(f)
    nonzero_df = df != 0
    idf[nonzero_df] = np.log(s / df[nonzero_df])

    # Compute TF-IDF
    for i, sentence in enumerate(tokenized):
        sentence_len = len(sentence)

        if sentence_len == 0:
            continue

        for j, word in enumerate(features):
            tf = sentence.count(word) / sentence_len
            embeddings[i, j] = tf * idf[j]

    return embeddings, features
