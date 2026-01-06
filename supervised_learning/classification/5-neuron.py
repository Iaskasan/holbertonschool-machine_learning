#!/usr/bin/env python3
"""Clas neuron module"""
import numpy as np


class Neuron:
    """Neuron class that defines a single
    neuron performing binary classification"""

    def __init__(self, nx):
        """Constructor method

        Args:
            nx (int): number of input features to the neuron
            W (numpy.ndarray): weights vector
            b (float): bias value
            A (float): activated output
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Weights getter

        Returns:
            numpy.ndarray: weights vector
        """
        return self.__W

    @property
    def b(self):
        """Bias getter

        Returns:
            float: bias value
        """
        return self.__b

    @property
    def A(self):
        """Activated output getter

        Returns:
            float: activated output
        """
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron

        Args:
            X (numpy.ndarray): input data of shape (nx, m)

        Returns:
            numpy.ndarray: activated output of the neuron
        """
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Args:
            Y (numpy.ndarray): correct labels of shape (1, m)
            A (numpy.ndarray): activated output of shape (1, m)
        Returns:
            float: cost of the model
        """
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(Y * np.log(A) +
                                  (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions

        Args:
            X (numpy.ndarray): input data of shape (nx, m)
            Y (numpy.ndarray): correct labels of shape (1, m)

        Returns:
            tuple: predicted labels and cost of the neuron
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = np.where(A >= 0.5, 1, 0)
        return predictions, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Performs one pass of gradient descent on the neuron

        Args:
            X (numpy.ndarray): input data of shape (nx, m)
            Y (numpy.ndarray): correct labels of shape (1, m)
            A (numpy.ndarray): activated output of shape (1, m)
            alpha (float): learning rate
        """
        m = Y.shape[1]
        dZ = A - Y
        dW = (1 / m) * np.dot(dZ, X.T)
        db = (1 / m) * np.sum(dZ)
        self.__W -= alpha * dW
        self.__b -= alpha * db
