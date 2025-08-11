#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
"""Simple line"""


def line():
    """Generates a line graph"""
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))
    x = np.arange(0, 11)
    plt.plot(x, y, color="red")
    plt.xlim(0, 10)
    plt.show()
