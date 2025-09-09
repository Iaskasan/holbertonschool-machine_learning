#!/usr/bin/env python3
"""Update var via Adam algo"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable using the Adam optimization algorithm.

    Args:
        alpha (float): learning rate
        beta1 (float): weight for first moment
        beta2 (float): weight for second moment
        epsilon (float): small constant to avoid division by zero
        var (np.ndarray): variable to be updated
        grad (np.ndarray): gradient of var
        v (np.ndarray): previous first moment
        s (np.ndarray): previous second moment
        t (int): time step (for bias correction)

    Returns:
        var (np.ndarray): updated variable
        v (np.ndarray): new first moment
        s (np.ndarray): new second moment
    """
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    v_corrected = v / (1 - beta1**t)
    s_corrected = s / (1 - beta2**t)

    var = var - alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)

    return var, v, s
