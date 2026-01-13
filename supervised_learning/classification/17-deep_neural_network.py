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
