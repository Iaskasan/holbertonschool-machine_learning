#!/usr/bin/env python3
"""Performs K-means on a dataset"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means.

    Args:
        X: numpy.ndarray of shape (n, d)
        k: positive integer number of clusters

    Returns:
        numpy.ndarray of shape (k, d) containing initialized centroids,
        or None on failure.
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(k, int) or k <= 0):
        return None

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)

    return np.random.uniform(low=low, high=high, size=(k, X.shape[1]))


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d)
        k: positive integer number of clusters
        iterations: positive integer maximum number of iterations

    Returns:
        C, clss
        C is a numpy.ndarray of shape (k, d) containing centroids
        clss is a numpy.ndarray of shape (n,) containing cluster indices

        Returns (None, None) on failure.
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(k, int) or k <= 0 or
            not isinstance(iterations, int) or iterations <= 0):
        return None, None

    n, d = X.shape

    if k > n:
        return None, None

    C = initialize(X, k)
    if C is None:
        return None, None

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)

    for _ in range(iterations):
        old_C = C.copy()

        # Compute distances from each point to each centroid
        distances = np.linalg.norm(X[:, np.newaxis, :] - C[np.newaxis, :, :],
                                   axis=2)

        # Assign each point to nearest centroid
        clss = np.argmin(distances, axis=1)

        # Update centroids
        for j in range(k):
            points = X[clss == j]

            if points.shape[0] == 0:
                C[j] = np.random.uniform(low=low, high=high, size=(d,))
            else:
                C[j] = np.mean(points, axis=0)

        if np.array_equal(C, old_C):
            return C, clss

    return C, clss
