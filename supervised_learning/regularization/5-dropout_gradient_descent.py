#!/usr/bin/env python3
"""
Gradient descent with Dropout for a neural network
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization
    using gradient descent.

    Args:
        Y: np.ndarray of shape (classes, m) - true one-hot labels
        weights: dict of weights and biases of the network (updated in-place)
        cache: dict of activations and dropout masks
        alpha: learning rate
        keep_prob: probability that a node will be kept
        L: number of layers
    """
    m = Y.shape[1]

    # Initialize dA for the output layer (softmax + cross-entropy)
    A_L = cache['A' + str(L)]
    dA = A_L - Y  # derivative of softmax cross-entropy

    for i in reversed(range(1, L + 1)):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]

        # Linear gradients
        dW = (1 / m) * np.dot(dA, A_prev.T)
        db = (1 / m) * np.sum(dA, axis=1, keepdims=True)

        # Update weights and biases in-place
        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db

        if i > 1:
            # Backpropagate to previous layer
            dA_prev = np.dot(W.T, dA)

            # Apply dropout mask
            D_prev = cache['D' + str(i - 1)]
            dA_prev *= D_prev
            dA_prev /= keep_prob

            # Apply tanh derivative
            A_prev_hidden = cache['A' + str(i - 1)]
            dA = dA_prev * (1 - A_prev_hidden ** 2)
