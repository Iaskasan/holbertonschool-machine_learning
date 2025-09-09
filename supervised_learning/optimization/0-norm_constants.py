#!/usr/bin/env python3
import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization (standardization) constants of a matrix

    Args:
        X (np.ndarray): shape (m, nx), the data to normalize

    Returns:
        mean (np.ndarray): shape (nx,), mean of each feature
        std (np.ndarray): shape (nx,), standard deviation of each feature
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
