#!/usr/bin/env python3
"""Class NauralNetwork module"""
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """Neuron class that defines a single
    neuron performing binary classification"""

    def __init__(self, nx, nodes):
        """Constructor method
        Args:
            nx (int): number of input features to the neuron
            nodes (int): number of nodes found in the hidden layer
            W1 (numpy.ndarray): weights vector for the hidden layer
            b1 (numpy.ndarray): bias vector for the hidden layer
            A1 (numpy.ndarray): activated output for the hidden layer
            W2 (numpy.ndarray): weights vector for the output neuron
            b2 (float): bias for the output neuron
            A2 (float): activated output for the output neuron
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.nx = nx
        self.nodes = nodes
        self.W1 = np.random.normal(size=(nodes, nx))
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.normal(size=(1, nodes))
        self.b2 = 0
        self.A2 = 0
