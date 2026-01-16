#!/usr/bin/env python3
"""one_hot_encode module"""
import numpy as np


def one_hot_encode(Y, classes):
    """one_hot_encode - one hot encodes a numeric label vector
    Args:
        Y (numpy.ndarray): shape (m,) that contains the numeric labels
        classes (int): number of classes
    Returns:
        numpy.ndarray: shape (classes, m) containing the one-hot
        encoding of Y
    """
    m = Y.shape[0]
    one_hot = np.zeros((classes, m))
    for i in range(m):
        one_hot[Y[i], i] = 1
    return one_hot
