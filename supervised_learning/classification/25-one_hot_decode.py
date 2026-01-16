#!/usr/bin/env python3
"""one_hot_decode module"""
import numpy as np


def one_hot_decode(one_hot):
    """one_hot_decode - decodes a one-hot encoded matrix
    Args:
        one_hot (numpy.ndarray): shape (classes, m) containing
        the one-hot encoding of Y
    Returns:
        numpy.ndarray: shape (m,) containing the numeric labels
        for each example
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None
    classes, m = one_hot.shape
    Y_decoded = np.zeros((m,), dtype=int)
    for i in range(m):
        Y_decoded[i] = np.argmax(one_hot[:, i])
    return Y_decoded
