#!/usr/bin/env python3
"""Performs K-means on a dataset"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset.

    X is a numpy.ndarray of shape (n, d) containing the dataset
    k is a positive integer containing the number of clusters
    iterations is a positive integer containing the maximum number
    of iterations that should be performed

    Returns:
    C, clss
    C is a numpy.ndarray of shape (k, d) containing the centroid means
    clss is a numpy.ndarray of shape (n,) containing the index of the
    cluster in C that each data point belongs to

    Returns (None, None) on failure
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
        # Assign each point to closest centroid
        distances = np.linalg.norm(X[:, np.newaxis, :] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        C_new = np.copy(C)

        # Update centroids
        for j in range(k):
            points = X[clss == j]

            if points.shape[0] == 0:
                C_new[j] = np.random.uniform(low=low, high=high, size=(d,))
            else:
                C_new[j] = np.mean(points, axis=0)

        # Stop if no centroid changed
        if np.array_equal(C, C_new):
            return C, clss

        C = C_new

    # Recompute classes for the final centroids
    distances = np.linalg.norm(X[:, np.newaxis, :] - C, axis=2)
    clss = np.argmin(distances, axis=1)

    return C, clss
