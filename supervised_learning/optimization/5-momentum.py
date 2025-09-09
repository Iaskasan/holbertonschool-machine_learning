#!/usr/bin/env python3
"""Gradient descent with momentum"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using gradient descent with momentum.

    Args:
        alpha (float): learning rate
        beta1 (float): momentum weight (0 < beta1 < 1)
        var (np.ndarray): variable to be updated
        grad (np.ndarray): gradient of var
        v (np.ndarray): previous first moment (velocity)

    Returns:
        var (np.ndarray): updated variable
        v (np.ndarray): new velocity (first moment)
    """
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v
    return var, v
