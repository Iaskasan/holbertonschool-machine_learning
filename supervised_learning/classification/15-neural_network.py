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
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """
        Performs forward propagation through the neural network.

        Steps:
            1. Compute Z1 = W1·X + b1   (linear combination for hidden layer)
            2. Apply sigmoid activation → A1
            3. Compute Z2 = W2·A1 + b2  (linear combination for output layer)
            4. Apply sigmoid activation → A2

        Args:
            X (numpy.ndarray): input data of shape (nx, m)
                nx = number of input features
                m = number of examples

        Updates:
            __A1 (numpy.ndarray): hidden layer activations, shape (nodes, m)
            __A2 (numpy.ndarray): output layer activations (predictions),
            shape (1, m)

        Returns:
            tuple: (__A1, __A2)
                __A1: hidden layer activations, values in (0, 1)
                __A2: output layer predictions, values in (0, 1)
        """
        Z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        "Calculate the cost of a model using logistic regression"
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """evaluate the neuron's prediction"""
        _, A = self.forward_prop(X)
        # .astype transforms booleans into int(O or 1)
        pred = (A >= 0.5).astype(int)
        return pred, self.cost(Y, A)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        One pass of gradient descent for a
        1-hidden-layer NN (sigmoid -> sigmoid)

        Shapes:
        X  : (nx, m)
        Y  : (1, m)
        A1 : (nodes, m)
        A2 : (1, m)
        W1 : (nodes, nx)
        b1 : (nodes, 1)
        W2 : (1, nodes)
        b2 : (1, 1)
        """
        m = Y.shape[1]

        dZ2 = A2 - Y
        dW2 = (dZ2 @ A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dZ1 = (self.__W2.T @ dZ2) * (A1 * (1 - A1))
        dW1 = (dZ1 @ X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        self.__W1 = self.__W1 - alpha * dW1
        self.__W2 = self.__W2 - alpha * dW2
        self.__b1 = self.__b1 - alpha * db1
        self.__b2 = self.__b2 - alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """train the neuron"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        for _ in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
        return self.evaluate(X, Y)

