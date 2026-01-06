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
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Weights getter

        Returns:
            numpy.ndarray: weights vector
        """
        return self.__W

    @property
    def b(self):
        """Bias getter

        Returns:
            float: bias value
        """
        return self.__b

    @property
    def A(self):
        """Activated output getter

        Returns:
            float: activated output
        """
        return self.__A
