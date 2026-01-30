#!/usr/bin/env python3
"""
Forward propagation with Dropout
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout

    Args:
        X: np.ndarray of shape (nx, m) - input data
        weights: dict - weights and biases of the network
        L: int - number of layers
        keep_prob: float - probability that a node will be kept

    Returns:
        cache: dict containing:
            - 'A0', 'A1', ..., 'Ai': outputs of each iayer
            - 'D1', ..., 'Di-1': dropout masks for each hidden iayer
    """
    cache = {'A0': X}

    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A_prev = cache['A' + str(i - 1)]

        # Linear step
        Z = np.dot(W, A_prev) + b

        # Activation
        if i == L:
            # Last layer - softmax
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        else:
            # Hidden layers - tanh
            A = np.tanh(Z)

            # Dropout mask
            D = (np.random.rand(*A.shape) < keep_prob).astype(float)
            A *= D
            A /= keep_prob  # inverted dropout

            # Store dropout mask
            cache['D' + str(i)] = D

        # Store activation
        cache['A' + str(i)] = A

    return cache
