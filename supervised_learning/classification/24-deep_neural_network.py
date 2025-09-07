#!/usr/bin/env python3
"""deep neural network module"""
import numpy as np
import matplotlib.pyplot as plt


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

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m).
            Y (numpy.ndarray): Correct labels of shape (1, m).

        Returns:
            tuple:
                predictions (numpy.ndarray): Shape (1, m), predicted labels
                    where values are 1 if output >= 0.5, else 0.
                cost (float): The logistic regression cost of the network.
        """
        AL, _ = self.forward_prop(X)
        predictions = (AL >= 0.5).astype(int)
        cost = self.cost(Y, AL)

        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Performs one pass of gradient descent on the deep neural network
        with sigmoid activations in all layers.

        Args:
            Y (np.ndarray): True labels, shape (1, m).
            cache (dict): Cached activations from forward_prop.
                Must contain: A0 (X), A1, ..., AL
            alpha (float): Learning rate (> 0).

        Updates:
            self.weights: in-place updates of Wl and bl for l = 1..L

        Notes:
            - One loop over layers (from L down to 1).
            - Uses only activations A0..AL from cache (no need to store Z).
            - For sigmoid, g'(Z) = A*(1-A).
        """
        m = Y.shape[1]
        last = self.L
        AL = cache[f"A{last}"]
        dZ = AL - Y

        for layer_idx in range(last, 0, -1):
            A_prev = cache[f"A{layer_idx - 1}"]
            W_curr = self.weights[f"W{layer_idx}"]
            b_curr = self.weights[f"b{layer_idx}"]

            dW = (dZ @ A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            if layer_idx > 1:
                A_curr = cache[f"A{layer_idx - 1}"]
                dZ_prev = (W_curr.T @ dZ) * (A_curr * (1 - A_curr))

            self.weights[f"W{layer_idx}"] = W_curr - alpha * dW
            self.weights[f"b{layer_idx}"] = b_curr - alpha * db

            if layer_idx > 1:
                dZ = dZ_prev

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the deep neural network by gradient descent.

        Args:
            X (np.ndarray): input data of shape (nx, m)
            Y (np.ndarray): true labels of shape (1, m)
            iterations (int): number of training iterations (>= 1)
            alpha (float): learning rate (> 0)
            verbose (bool): if True, print cost every `step` iterations
                (incl. 0 and last)
            graph (bool): if True, plot cost vs iteration after training
            step (int): interval for verbose/graph; must be 1..iterations
                (inclusive) when verbose or graph is True

        Raises (in order):
            TypeError: iterations must be an integer
            ValueError: iterations must be a positive integer
            TypeError: alpha must be a float
            ValueError: alpha must be positive
            TypeError: step must be an integer (only if verbose or graph)
            ValueError: step must be positive and <= iterations
                (only if verbose or graph)

        Updates:
            - self.__cache via forward_prop (A0..AL)
            - self.__weights via gradient_descent (Wl, bl)

        Returns:
            tuple: (predictions, cost) from evaluate(X, Y) after training.
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        it_hist = []
        cost_hist = []

        for i in range(iterations + 1):
            AL, cache = self.forward_prop(X)

            should_log = (i == 0) or (i % step == 0) or (i == iterations)
            if should_log:
                J = self.cost(Y, AL)
                if verbose:
                    print(f"Cost after {i} iterations: {J}")
                if graph:
                    it_hist.append(i)
                    cost_hist.append(J)

            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph:
            plt.plot(it_hist, cost_hist)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix.

    Args:
        Y (np.ndarray): shape (m,) with numeric class labels
        classes (int): total number of classes

    Returns:
        np.ndarray of shape (classes, m): one-hot encoded matrix
        or None if input validation fails
    """
    if not isinstance(Y, np.ndarray) or not isinstance(classes, int):
        return None
    if Y.ndim != 1:
        return None
    if classes <= np.max(Y):
        return None

    m = Y.shape[0]
    one_hot = np.zeros((classes, m))

    one_hot[Y, np.arange(m)] = 1

    return one_hot
