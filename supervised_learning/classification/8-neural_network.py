#!/usr/bin/env python3
"""Neural network module"""
import numpy as np


class NeuralNetwork:
    """Neural Network class"""
    def __init__(self, nx, nodes):
        """
        Initialize a NeuralNetwork with one hidden layer.

        Args:
            nx (int): Number of input features.
                Must be a positive integer.
            nodes (int): Number of nodes in the hidden layer.
                Must be a positive integer.

        Attributes:
            W1 (numpy.ndarray): Weights for the hidden layer,
                shape (nodes, nx), initialized with random normal values.
            b1 (numpy.ndarray): Biases for the hidden layer,
                shape (nodes, 1), initialized with zeros.
            A1 (float): Activated output for the hidden layer,
                initialized to 0.

            W2 (numpy.ndarray): Weights for the output neuron,
                shape (1, nodes), initialized with random normal values.
            b2 (float): Bias for the output neuron, initialized to 0.
            A2 (float): Activated output (prediction) for the output neuron,
                initialized to 0.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.nodes = nodes
        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0

        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
