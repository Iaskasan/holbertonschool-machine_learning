#!/usr/bin/env python3
"""Class NauralNetwork module"""
import numpy as np


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
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Weights getter for hidden layer

        Returns:
            numpy.ndarray: weights vector for the hidden layer
        """
        return self.__W1

    @property
    def b1(self):
        """Bias getter for hidden layer

        Returns:
            numpy.ndarray: bias vector for the hidden layer
        """
        return self.__b1

    @property
    def A1(self):
        """Activated output getter for hidden layer

        Returns:
            numpy.ndarray: activated output for the hidden layer
        """
        return self.__A1

    @property
    def W2(self):
        """Weights getter for output neuron

        Returns:
            numpy.ndarray: weights vector for the output neuron
        """
        return self.__W2

    @property
    def b2(self):
        """Bias getter for output neuron

        Returns:
            float: bias for the output neuron
        """
        return self.__b2

    @property
    def A2(self):
        """Activated output getter for output neuron

        Returns:
            float: activated output for the output neuron
        """
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network

        Args:
            X (numpy.ndarray): input data of shape (nx, m)

        Returns:
            numpy.ndarray: activated output of the output neuron
        """
        Z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2
