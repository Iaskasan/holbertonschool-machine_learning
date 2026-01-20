#!/usr/bin/env python3
"""calculates the weighted moving average of a data set"""
import numpy as np
import matplotlib.pyplot as plt


def moving_average(data, beta):
    """calculates moving average of a dataset
    Args:
        data (list): data to calculate the moving average of
        beta (float): weight used for the moving average
    Returns:
        list: moving averages of data
    """
    ema = []
    ema.append(data[0])
    for i in range(1, len(data)):
        ema.append(beta * ema[i - 1] + (1 - beta) * data[i])
    return ema
