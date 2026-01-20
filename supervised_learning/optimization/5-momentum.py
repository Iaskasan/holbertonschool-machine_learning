#!/usr/bin/env python3
"""calculates the weighted moving average of a data set"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """updates a variable using the momentum optimization algorithm
    Args:
        alpha (float): learning rate
        beta1 (float): momentum weight
        var (numpy.ndarray): variable to be updated
        grad (numpy.ndarray): gradient of var
        v (numpy.ndarray): previous moment of var
    Returns:
        tuple: updated variable and the new moment
    """
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v
    return var, v
