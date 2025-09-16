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

    A_L = cache["A" + str(L)]
    dZ = A_L - Y

    for layer in range(L, 0, -1):
        A_prev = cache["A" + str(layer - 1)]
        W = weights["W" + str(layer)]

        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        weights["W" + str(layer)] -= alpha * dW
        weights["b" + str(layer)] -= alpha * db

        if layer > 1:
            A_prev = cache["A" + str(layer - 1)]
            dZ = np.matmul(W.T, dZ) * (1 - A_prev**2)
