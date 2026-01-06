#!/usr/bin/env python3
"""Clas neuron module"""
import numpy as np


class Neuron:
    """Neuron class that defines a single
    neuron performing binary classification"""

    def __init__(self, nx):
        """Constructor method

        Args:
            nx (int): number of input features to the neuron
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        self.W = np.random.normal((1, nx))
        self.b = 0
        self.A = 0
