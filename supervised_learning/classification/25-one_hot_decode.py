#!/usr/bin/env python3
"""one hot decode method"""
import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels.

    Args:
        one_hot (np.ndarray): one-hot encoded array
            shape (classes, m)

    Returns:
        np.ndarray of shape (m,): numeric class labels
        or None if input validation fails
    """
    if not isinstance(one_hot, np.ndarray):
        return None
    if one_hot.ndim != 2:
        return None

    labels = np.argmax(one_hot, axis=0)
    return labels
