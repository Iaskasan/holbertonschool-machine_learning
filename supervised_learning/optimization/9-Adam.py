#!/usr/bin/env python3
"""updates a variable in place using the Adam optimization algorithm"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """updates a variable in place using the Adam optimization algorithm
    Args:
        alpha (float): learning rate
        beta1 (float): momentum weight
        beta2 (float): RMSProp weight
        epsilon (float): small number to avoid division by zero
        var (numpy.ndarray): variable to be updated
        grad (numpy.ndarray): gradient of var
        v (numpy.ndarray): previous first moment of var
        s (numpy.ndarray): previous second moment of var
        t (int): time step used for bias correction
    Returns:
        tuple: updated variable, new first moment, and new second moment
    """
    v = beta1 * v + (1 - beta1) * grad
    v_corrected = v / (1 - beta1 ** t)
    s = beta2 * s + (1 - beta2) * (grad ** 2)
    s_corrected = s / (1 - beta2 ** t)
    var = var - alpha * v_corrected / (s_corrected ** 0.5 + epsilon)
    return var, v, s
