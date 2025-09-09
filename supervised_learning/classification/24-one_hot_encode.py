#!/usr/bin/env python3
"""one hot encode method"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix.

    Args:
        Y (np.ndarray): shape (m,) with numeric class labels
        classes (int): total number of classes

    Returns:
        np.ndarray of shape (classes, m): one-hot encoded matrix
        or None if input validation fails
    """
    if not isinstance(Y, np.ndarray) or not isinstance(classes, int):
        return None
    if Y.ndim != 1:
        return None
    if classes <= np.max(Y):
        return None

    m = Y.shape[0]
    one_hot = np.zeros((classes, m))

    one_hot[Y, np.arange(m)] = 1

    return one_hot
