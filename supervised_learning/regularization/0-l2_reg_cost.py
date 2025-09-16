#!/usr/bin/env python3
"""L2 Regularization Cost"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization

    Parameters
    ----------
    cost : float
        The cost of the network without L2 regularization (e.g. cross-entropy).
    lambtha : float
        The regularization parameter (λ), controls the strength of the penalty.
    weights : dict
        Dictionary of weights and biases of the neural network.
        Keys look like 'W1', 'b1', 'W2', 'b2', ... up to layer L.
    L : int
        Number of layers in the neural network.
    m : int
        Number of training examples.

    Returns
    -------
    float
        The cost of the network accounting for L2 regularization.
    """
    l2_sum = 0
    for i in range(1, L + 1):
        W = weights["W" + str(i)]
        l2_sum += np.sum(np.square(W))
    l2_cost = cost + (lambtha / (2 * m)) * l2_sum
    return l2_cost
