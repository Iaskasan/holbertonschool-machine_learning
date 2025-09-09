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
    X_shuffle = np.random.permutation(X)
    Y_shuffle = np.random.permutation(Y)
    return X_shuffle, Y_shuffle
