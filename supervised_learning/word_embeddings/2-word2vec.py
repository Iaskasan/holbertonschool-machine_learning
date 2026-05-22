#!/usr/bin/env python3
"""Word2Vec model creation and training."""

import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Create, build, and train a gensim Word2Vec model.

    Args:
        sentences: list of sentences to train on
        vector_size: dimensionality of the embeddings
        min_count: minimum word count threshold
        window: maximum distance between current and predicted word
        negative: number of negative samples
        cbow: True for CBOW, False for Skip-gram
        epochs: number of training iterations
        seed: random seed
        workers: number of worker threads

    Returns:
        A trained Word2Vec model
    """

    model = gensim.models.Word2Vec(
        sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=0 if cbow else 1,
        epochs=epochs,
        seed=seed,
        workers=workers
    )

    return model
