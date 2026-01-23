#!/usr/bin/env python3
"""updates a variable using the RMSProp optimization algorithm"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """updates a variable using the RMSProp optimization algorithm
    Args:
        alpha (float): learning rate
        beta2 (float): RMSProp weight
        epsilon (float): small number to avoid division by zero
        var (numpy.ndarray): variable to be updated
        grad (numpy.ndarray): gradient of var
        s (numpy.ndarray): previous squared gradient of var
    Returns:
        tuple: updated variable and the new squared gradient
    """
    s = beta2 * s + (1 - beta2) * (grad ** 2)
    var = var - alpha * grad / (s ** 0.5 + epsilon)
    return var, s
