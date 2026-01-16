#!/usr/bin/env python3
"""Class DeepNeuralNetwork module"""
import numpy as np
import matplotlib.pyplot as plt


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

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions
        Args:
            X (numpy.ndarray): shape (nx, m) that contains the input data
            nx (int): number of input features to the neuron
            m (int): number of examples
            Y (numpy.ndarray): shape (1, m) that contains the correct
            labels for the input data
        Returns:
            the neuron’s prediction and the cost of the network
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient
        descent on the neural network
        Args:
            Y (numpy.ndarray): shape (1, m) that contains the correct
            labels for the input data
            cache (dict): dictionary containing all
            intermediary values of the network
            alpha (float): learning rate
        """
        m = Y.shape[1]
        dZ = cache[f"A{self.L}"] - Y
        for i in range(self.L, 0, -1):
            A_prev = cache[f"A{i - 1}"]
            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            if i > 1:
                dZ = np.matmul(
                    self.weights[f"W{i}"].T, dZ) * A_prev * (1 - A_prev)
            self.weights[f"W{i}"] -= alpha * dW
            self.weights[f"b{i}"] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the deep neural network
        Args:
            X (numpy.ndarray): shape (nx, m) that contains the input data
            nx (int): number of input features to the neuron
            m (int): number of examples
            Y (numpy.ndarray): shape (1, m) that contains the correct
            labels for the input data
            iterations (int): number of iterations to train over
            alpha (float): learning rate
            verbose (bool): boolean that defines whether or not to print
            information about the training
            graph (bool): boolean that defines whether or not to graph
            information about the training once the training has completed
            step (int): number of iterations between printing and graphing
        Returns:
            the evaluation of the training data after iterations of training
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        iteration_steps = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A)
                costs.append(cost)
                iteration_steps.append(i)
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")
            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph:
            plt.plot(iteration_steps, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format
        Args:
            filename (str): file to which the object should be saved
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object
        Args:
            filename (str): file from which the object should be loaded
        Returns:
            the loaded object, or None if filename doesn't exist
        """
        import pickle
        try:
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
            return obj
        except FileNotFoundError:
            return None
