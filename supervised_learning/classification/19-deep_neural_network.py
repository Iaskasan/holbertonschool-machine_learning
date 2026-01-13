#!/usr/bin/env python3
"""Class DeepNeuralNetwork module"""
import numpy as np


class DeepNeuralNetwork:
    """DeepNeuralNetwork class that defines a single
    neuron performing binary classification"""

    def __init__(self, nx, layers):
        """Constructor method
        Args:
            nx (int): number of input features to the neuron
            layers (list): list representing the number of nodes
            found in each layer of the network
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        prev = nx
        # dictionary to hold all weights and biases of the network
        self.__weights = {}
        for i, nodes in enumerate(layers, start=1):
            if not isinstance(nodes, int) or nodes < 1:
                raise TypeError("layers must be a list of positive integers")

            self.weights[f"W{i}"] = np.random.randn(nodes,
                                                    prev) * np.sqrt(2 / prev)
            self.weights[f"b{i}"] = np.zeros((nodes, 1))

            prev = nodes
        self.nx = nx
        self.layers = layers
        # number of layers in the neural network
        self.__L = len(layers)
        # dictionary to hold all intermediary values of the network
        self.__cache = {}

    @property
    def L(self):
        """Layers getter

        Returns:
            int: number of layers in the neural network
        """
        return self.__L

    @property
    def cache(self):
        """Cache getter

        Returns:
            dict: dictionary to hold all intermediary values of the network
        """
        return self.__cache

    @property
    def weights(self):
        """Weights getter

        Returns:
            dict: dictionary to hold all weights and biases of the network
        """
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the deep neural network
        Args:
            X (numpy.ndarray): shape (nx, m) that contains the input data
            nx (int): number of input features to the neuron
            m (int): number of examples
        Returns:
            the output of the neural network and the cache
        """
        self.cache["A0"] = X
        for i in range(1, self.L + 1):
            W = self.weights[f"W{i}"]
            b = self.weights[f"b{i}"]
            A_prev = self.cache[f"A{i - 1}"]

            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))

            self.cache[f"A{i}"] = A

        return A, self.cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression
        Args:
            Y (numpy.ndarray): shape (1, m) that contains the correct
            labels for the input data
            A (numpy.ndarray): shape (1, m) containing the activated output
            of the neuron for each example
            m (int): number of examples
        Returns:
            the cost
        """
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(Y * np.log(A) +
                                  (1 - Y) * np.log(1.0000001 - A))
        return cost
