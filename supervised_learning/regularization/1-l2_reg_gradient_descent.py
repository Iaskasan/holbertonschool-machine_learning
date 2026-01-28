#!/usr/bin/env python3
"""
updates the weights and biases of a neural
network using gradient descent with L2 regularization
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using
    gradient descent with L2 regularization

    Args:
        Y: numpy.ndarray - one-hot numpy.ndarray of shape (classes, m)
                           with correct labels
        weights: dict - weights and biases (keys 'W1', 'b1', ...)
        cache: dict - outputs of each layer (keys 'A0', 'A1', ...)
        alpha: float - learning rate
        lambtha: float - regularization parameter
        L: int - number of layers

    Returns:
        None - updates weights in place
    """
    m = Y.shape[1]
    dZ = cache["A" + str(L)] - Y

    for i in reversed(range(1, L + 1)):
        A_prev = cache["A" + str(i - 1)]
        W = weights["W" + str(i)]
        b = weights["b" + str(i)]

        dW = (1 / m) * (dZ @ A_prev.T) + (lambtha / m) * W
        db = (1 / m) * dZ.sum(axis=1, keepdims=True)

        weights["W" + str(i)] = W - alpha * dW
        weights["b" + str(i)] = b - alpha * db
        if i > 1:
            A_prev = cache["A" + str(i - 1)]
            dZ = (W.T @ dZ) * (1 - A_prev ** 2)
