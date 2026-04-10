#!/usr/bin/env python3
"""Performs K-means on a dataset"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset.

    Returns:
        C, clss
    or:
        None, None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    low = np.min(X, axis=0)
    high = np.max(X, axis=0)

    C = np.random.uniform(low=low, high=high, size=(k, d))

    for _ in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis, :] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        C_new = np.copy(C)

        for j in range(k):
            points = X[clss == j]

            if points.shape[0] == 0:
                C_new[j] = np.random.uniform(low=low, high=high, size=(d,))
            else:
                C_new[j] = np.mean(points, axis=0)

        if np.array_equal(C, C_new):
            return C, clss

        C = C_new

    distances = np.linalg.norm(X[:, np.newaxis, :] - C, axis=2)
    clss = np.argmin(distances, axis=1)

    return C, clss
