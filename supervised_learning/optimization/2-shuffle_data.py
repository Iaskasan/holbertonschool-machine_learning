#!/usr/bin/env python3
"""shuffle dataset X
"""
import numpy as np


def shuffle_data(X, Y):
    """shuffles data points in X and Y in the same way
    """
    X_shuffle = np.random.permutation(X.shape[0])
    return X[X_shuffle], Y[X_shuffle]
