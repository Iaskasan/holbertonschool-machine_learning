#!/usr/bin/env python3
"""Class DeepNeuralNetwork module"""
import numpy as np


class DeepNeuralNetwork:
    """DeeNeuralNetwork class that defines a single
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
        for elements in layers:
            if not isinstance(elements, int) or elements < 1:
                raise TypeError("layers must be a list of positive integers")
        self.nx = nx
        self.layers = layers
        self.L = len(layers) # number of layers in the neural network
        self.cache = {}      # dictionary to hold all intermediary values of the network
        self.weights = {}    # dictionary to hold all weights and biases of the network
        for l in range(self.L):
            if l == 0:
                self.weights['W' + str(l + 1)] = np.random.randn(
                    layers[l], nx) * np.sqrt(2 / nx)
            else:
                self.weights['W' + str(l + 1)] = np.random.randn(
                    layers[l], layers[l - 1]) * np.sqrt(2 / layers[l - 1])
            self.weights['b' + str(l + 1)] = np.zeros((layers[l], 1))
