#!/usr/bin/env python3
"""calculates the weighted moving average of a data set"""


def moving_average(data, beta):
    """calculates moving average of a dataset
    Args:
        data (list): data to calculate the moving average of
        beta (float): weight used for the moving average
    Returns:
        list: moving averages of data
    """
    ema = []
    ema_val = 0
    for t, x in enumerate(data):
        ema_val = beta * ema_val + (1 - beta) * x
        ema.append(ema_val / (1 - beta**(t+1)))
    return ema
