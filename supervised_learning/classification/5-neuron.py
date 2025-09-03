#!/usr/bin/env python3
"""Neuron module"""
import numpy as np


class Neuron:
    """Neuron class"""
    def __init__(self, nx):
        """
        nx: number of input features to the neuron
        __W: weights of the neuron
        __b: bias
        __A: activated output (prediction)
        """
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

    def forward_prop(self, X):
        """Perform forward propagation of the neuron"""
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        "Calculate the cost of a model using logistic regression"
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """evaluate the neuron's prediction"""
        A = self.forward_prop(X)
        # .astype transforms booleans into int(O or 1)
        pred = (A >= 0.5).astype(int)
        return pred, self.cost(Y, A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """calculate one pass of gradient descent on the neuron"""
        m = Y.shape[1]
        dZ = A - Y
        dW = (dZ @ X.T) / m
        db = np.sum(dZ) / m
        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db
