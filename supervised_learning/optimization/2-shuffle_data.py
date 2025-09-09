#!/usr/bin/env python3
"""Shuffle data"""
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles two matrices the same way

    Args:
        X (np.ndarray): shape (m, nx), first dataset
        Y (np.ndarray): shape (m, ny), second dataset

    Returns:
        (X_shuffled, Y_shuffled): shuffled versions of X and Y
    """
    m = X.shape[0]
    shuffle = np.random.permutation(m)
    return X[shuffle], Y[shuffle]
