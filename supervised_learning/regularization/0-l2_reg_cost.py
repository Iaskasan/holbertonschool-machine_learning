#!/usr/bin/env python3
"""calculates the cost of a neural network with L2 regularization"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization

    Args:
        cost: float - cost without regularization
        lambtha: float - regularization parameter
        weights: dict - weights and biases (keys 'W1', 'b1', ...)
        L: int - number of layers
        m: int - number of examples

    Returns:
        float - cost with L2 regularization
    """
    l2_sum = 0
    for i in range(1, L+1):
        W = weights['W' + str(i)]
        l2_sum += np.sum(W**2)

    l2_cost = (lambtha / (2 * m)) * l2_sum
    total_cost = cost + l2_cost
    return total_cost
