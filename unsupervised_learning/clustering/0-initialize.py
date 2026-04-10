#!/usr/bin/env python3
"""Initializes cluster centroids for K-means"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means

    Parameters:
    X (np.ndarray): shape (n, d)
    k (int): number of clusters

    Returns:
    np.ndarray of shape (k, d) or None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape

    # Min and max per dimension
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    # Uniform initialization (vectorized, one call)
    centroids = np.random.uniform(
        low=min_vals,
        high=max_vals,
        size=(k, d)
    )

    return centroids
