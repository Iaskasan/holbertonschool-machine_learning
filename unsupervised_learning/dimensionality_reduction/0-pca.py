#!/usr/bin/env python3
"""Performs PCA on a dataset."""
import numpy as np


def pca(X, var=0.95):
    """Performs PCA on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d)
            n is the number of data points
            d is the number of dimensions
            Data is assumed to be centered.
        var: fraction of variance to preserve

    Returns:
        W: numpy.ndarray of shape (d, nd)
           projection matrix preserving at least var variance
    """
    _, S, Vt = np.linalg.svd(X, full_matrices=False)

    explained_var = S ** 2
    cumulative_var = np.cumsum(explained_var) / np.sum(explained_var)

    nd = np.searchsorted(cumulative_var, var) + 1

    W = Vt[:nd].T
    return W
