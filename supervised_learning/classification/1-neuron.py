#!/usr/bin/env python3
"""Neuron module"""
import numpy as np


class Neuron:
    """Neuron class"""
    def __init__(self, nx):
        """nx: number of input features to the neuron"""
        self.nx = nx
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A
