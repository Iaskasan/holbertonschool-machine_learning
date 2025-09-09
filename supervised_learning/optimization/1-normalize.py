#!/usr/bin/env python3
"""standardization of a matrix"""


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix

    Args:
        X (np.ndarray): shape (d, nx), data to normalize
        m (np.ndarray): shape (nx,), mean of each feature
        s (np.ndarray): shape (nx,), std of each feature

    Returns:
        np.ndarray: normalized matrix
    """
    return (X - m) / s
