#!/usr/bin/env python3
"""L2 Regularization Gradient Descent"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network
    using gradient descent with L2 regularization.

    Parameters
    ----------
    Y : np.ndarray
        One-hot array of shape (classes, m) with true labels.
    weights : dict
        Dictionary of weights and biases {W1, b1, W2, b2, ...}.
    cache : dict
        Dictionary of activations {A0, A1, ..., AL}.
    alpha : float
        Learning rate.
    lambtha : float
        Regularization parameter.
    L : int
        Number of layers in the network.

    Notes
    -----
    Updates are done in place.
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for i in reversed(range(1, L + 1)):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        dW = (1 / m) * (np.matmul(dZ, A_prev.T)) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        weights['W' + str(i)] = W - alpha * dW
        weights['b' + str(i)] = weights['b' + str(i)] - alpha * db
        if i > 1:
            A_prev = cache['A' + str(i - 1)]
            dZ = np.matmul(W.T, dZ) * (1 - A_prev ** 2)
