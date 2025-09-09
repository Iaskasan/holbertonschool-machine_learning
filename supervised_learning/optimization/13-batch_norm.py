#!/usr/bin/env python3
"""perform a batch normalization before
passing to the activation function"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes the unactivated output of a neural
    network using batch normalization.

    Args:
        Z (np.ndarray): shape (m, n), raw outputs (m samples, n features).
        gamma (np.ndarray): shape (1, n), scale parameters.
        beta (np.ndarray): shape (1, n), shift parameters.
        epsilon (float): small constant to avoid division by zero.

    Returns:
        np.ndarray: normalized Z, same shape as input.
    """
    mean = np.mean(Z, axis=0, keepdims=True)
    var = np.var(Z, axis=0, keepdims=True)
    Z_hat = (Z - mean) / np.sqrt(var + epsilon)
    Z_norm = gamma * Z_hat + beta
    return Z_norm
