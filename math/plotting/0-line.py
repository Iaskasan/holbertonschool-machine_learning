#!/usr/bin/env python3
"""Generate a Simple line"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Plots a red line graph of y = x^3 for x values from 0 to 10.

    The function creates a figure and plots the cubic values
    of integers from 0 to 10,
    setting the x-axis limits from 0 to 10, and displays the plot.
    """
    """"""
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))
    x = np.arange(0, 11)
    plt.plot(x, y, color="red")
    plt.xlim(0, 10)
    plt.show()
