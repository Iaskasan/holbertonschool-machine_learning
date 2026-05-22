#!/usr/bin/env python3
"""FastText model creation and training."""

import gensim


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5,
                   window=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Create, build, and train a gensim FastText model.

    Args:
        sentences: list of sentences to train on
        vector_size: dimensionality of the embeddings
        min_count: minimum word count threshold
        negative: number of negative samples
        window: maximum distance between current and predicted word
        cbow: True for CBOW, False for Skip-gram
        epochs: number of training iterations
        seed: random seed
        workers: number of worker threads

    Returns:
        A trained FastText model
    """
    training_data = [
        sentence.split() if isinstance(sentence, str) else sentence
        for sentence in sentences
    ]

    model = gensim.models.FastText(
        training_data,
        vector_size=vector_size,
        min_count=min_count,
        negative=negative,
        window=window,
        sg=0 if cbow else 1,
        epochs=epochs,
        seed=seed,
        workers=workers
    )

    return model
