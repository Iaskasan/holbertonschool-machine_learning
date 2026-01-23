#!/usr/bin/env python3
"""normalizes an unactivated output
of a neural network using batch normalization"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """normalizes an unactivated output
    of a neural network using batch normalization

    Z: numpy.ndarray of shape (m, n) that
    contains the unactivated output of a neural network
        m: number of data points
        n: number of features
    gamma: numpy.ndarray of shape (1, n)
    containing the scales used for batch normalization
    beta: numpy.ndarray of shape (1, n)
    containing the offsets used for batch normalization
    epsilon: small number to avoid division by zero

    Returns: the normalized Z matrix
    """
    mu = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)
    Z_norm = (Z - mu) / np.sqrt(var + epsilon)
    Z_batch_norm = gamma * Z_norm + beta
    return Z_batch_norm
