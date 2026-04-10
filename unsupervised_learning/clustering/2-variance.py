#!/usr/bin/env python3
"""Calculates total intra-cluster variance"""
import numpy as np


def variance(X, C):
    """
    Calculates total intra-cluster variance

    Parameters:
    X (np.ndarray): shape (n, d)
    C (np.ndarray): shape (k, d)

    Returns:
    float: total variance or None on failure
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(C, np.ndarray) or C.ndim != 2 or
            X.shape[1] != C.shape[1]):
        return None

    # Compute squared distances
    distances = np.sum((X[:, np.newaxis, :] - C) ** 2, axis=2)

    # Minimum squared distance for each point
    min_dist = np.min(distances, axis=1)

    # Total variance
    return np.sum(min_dist)
