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

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression

        Args:
            Y (numpy.ndarray): correct labels for the input data
            A (numpy.ndarray): activated output of the neuron for each example

        Returns:
            float: cost of the model
        """
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(Y * np.log(A) +
                                  (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions

        Args:
            X (numpy.ndarray): input data of shape (nx, m)
            Y (numpy.ndarray): correct labels for the input data

        Returns:
            numpy.ndarray: neuron predictions
            float: cost of the model
        """
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        prediction = np.where(A2 >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient
        descent on the neural network
        Args:
            X (numpy.ndarray): input data of shape (nx, m)
            Y (numpy.ndarray): correct labels for the input data
            A1 (numpy.ndarray): activated output for the hidden layer
            A2 (numpy.ndarray): activated output for the output neuron
            alpha (float): learning rate
        """
        m = Y.shape[1]

        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.dot(self.__W2.T, dZ2) * A1 * (1 - A1)
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05,
            verbose=True, graph=True, step=100):
        """Trains the neural network

        Args:
            X (numpy.ndarray): input data of shape (nx, m)
            Y (numpy.ndarray): correct labels for the input data
            iterations (int): number of iterations to train over
            alpha (float): learning rate

        Returns:
            numpy.ndarray: neuron predictions
            float: cost of the model
        """

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if type(step) is not int:
            raise TypeError("step must be an integer")
        if step < 1 or step > iterations:
            raise ValueError("step must be positive and <= iterations")

        costs = []
        iters = []

        for i in range(iterations + 1):
            A1, A2 = self.forward_prop(X)
            cost = self.cost(Y, A2)

            if verbose and (i % step == 0 or i == iterations):
                print("Cost after {} iterations: {}".format(i, cost))

            if graph and i % step == 0:
                costs.append(cost)
                iters.append(i)

            if i < iterations:
                self.gradient_descent(X, Y, A1, A2, alpha)

        if graph:
            plt.plot(iters, costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
