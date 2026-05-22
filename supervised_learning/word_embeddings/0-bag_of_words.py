#!/usr/bin/env python3
"""Bag of Words embedding module."""

import re
import numpy as np


def tokenize(sentence):
    """Normalize and tokenize a sentence."""
    sentence = sentence.lower()
    sentence = re.sub(r"'s\b", "", sentence)
    sentence = re.sub(r"[^\w\s]", "", sentence)
    return sentence.split()


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

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

    embeddings = np.zeros((len(sentences), len(features)), dtype=int)

    for i, sentence in enumerate(tokenized):
        for j, word in enumerate(features):
            embeddings[i, j] = sentence.count(word)

    return embeddings, features
