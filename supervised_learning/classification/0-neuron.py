#!/usr/bin/env python3
"""Neuron module"""
import numpy as np


class Neuron:
    def __init__(self, nx):
        """nx: number of input features to the neuron"""
        self.nx = nx
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
