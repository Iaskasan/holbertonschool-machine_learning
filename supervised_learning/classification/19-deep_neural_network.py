#!/usr/bin/env python3
"""deep neural network module"""
import numpy as np


class DeepNeuralNetwork:
    """Deep Neural Network with arbitrary number of layers."""
    def __init__(self, nx, layers):
        """
        Initialize a deep neural network.

        Args:
            nx (int): Number of input features.
                Must be a positive integer.
            layers (list): List of positive integers where each integer
                represents the number of nodes in that layer.
                Example: [5, 3, 1] = 3-layer network with
                5 nodes in layer 1, 3 nodes in layer 2, 1 node in layer 3.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
            TypeError: If layers is not a list of positive integers.

        Public Attributes:
            L (int): Number of layers in the neural network
                     (length of layers list).
            cache (dict): Stores intermediary values of the network
                          (forward activations, etc.).
                          Initialized as an empty dictionary.
            weights (dict): Stores all weights and biases of the network:
                - Wl: Weight matrix for layer l
                      Shape: (nodes in layer l, nodes in layer l-1)
                      Initialized with He et al. method.
                - bl: Bias vector for layer l
                      Shape: (nodes in layer l, 1)
                      Initialized with zeros.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for layer in range(self.L):
            nodes_curr = layers[layer]
            if type(nodes_curr) is not int or nodes_curr <= 0:
                raise TypeError("layers must be a list of positive integers")
            nodes_prev = nx if layer == 0 else layers[layer - 1]
            self.__weights[f"W{layer+1}"] = (
                np.random.randn(nodes_curr, nodes_prev) *
                np.sqrt(2 / nodes_prev))
            self.__weights[f"b{layer+1}"] = np.zeros((nodes_curr, 1))

    @property
    def weights(self):
        return self.__weights

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    def forward_prop(self, X):
        """
        Performs forward propagation through the deep neural network.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m)
                nx = number of input features
                m  = number of examples

        Updates:
            self.cache (dict): Stores activations of each layer:
                - A0 = X
                - A1 = activation of layer 1
                - ...
                - AL = activation of output layer

        Returns:
            tuple:
                AL (numpy.ndarray): The output of the final layer, shape (1, m)
                self.cache (dict): The dictionary of all cached activations
        """
        self.cache["A0"] = X

        for layer in range(1, self.L + 1):
            Wl = self.weights[f"W{layer}"]
            bl = self.weights[f"b{layer}"]
            Al_prev = self.cache[f"A{layer-1}"]
            Zl = Wl @ Al_prev + bl
            Al = 1 / (1 + np.exp(-Zl))
            self.cache[f"A{layer}"] = Al
        return self.cache[f"A{self.L}"], self.cache

    def cost(self, Y, A):
        """
        Calculates the logistic regression cost.

        Args:
            Y (numpy.ndarray): True labels of shape (1, m).
            A (numpy.ndarray): Activated output (predictions) of shape (1, m).

        Returns:
            float: The cost value.
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost
