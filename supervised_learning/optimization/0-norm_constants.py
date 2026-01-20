#!/usr/bin/env python3
import numpy as np


def normalization_constants(X):
    """
    Compute normalization constants for each feature in dataset X.
    """
    mean = np.sum(X, axis=0) / X.shape[0]
    std = np.sqrt(np.sum(((X - mean) ** 2 / X.shape[0]), axis=0))
    return mean, std
