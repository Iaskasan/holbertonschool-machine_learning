#!/usr/bin/env python3
"""RMSProp adapts the learning rate for each parameter
based on the moving average of squared gradients,
so steep slopes get smaller steps and shallow slopes get larger steps"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using RMSProp optimization.

    Args:
        alpha (float): learning rate
        beta2 (float): RMSProp weight (0 < beta2 < 1)
        epsilon (float): small number to avoid division by zero
        var (np.ndarray): variable to be updated
        grad (np.ndarray): gradient of var
        s (np.ndarray): previous second moment of var

    Returns:
        var (np.ndarray): updated variable
        s (np.ndarray): new second moment
    """
    s = beta2 * s + (1 - beta2) * (grad ** 2)
    var = var - alpha * grad / (np.sqrt(s) + epsilon)
    return var, s
