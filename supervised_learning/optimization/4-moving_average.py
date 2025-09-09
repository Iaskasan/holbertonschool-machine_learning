#!/usr/bin/env python3
"""Exponential moving average"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a dataset with bias correction.

    Args:
        data (list): list of values
        beta (float): decay rate (0 < beta < 1)

    Returns:
        list: bias-corrected moving averages
    """
    averages = []
    v = 0
    for t, value in enumerate(data, 1):
        v = beta * v + (1 - beta) * value
        v_corrected = v / (1 - beta**t)
        averages.append(v_corrected)
    return averages
