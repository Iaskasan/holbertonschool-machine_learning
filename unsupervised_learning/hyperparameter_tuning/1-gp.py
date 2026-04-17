#!/usr/bin/env python3
"""Computes the optimal hyperparameters for a Gaussian Process"""
import numpy as np


class GaussianProcess():
    """Represents a noiseless 1D Gaussian process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Class constructor

        Args:
            X_init (numpy.ndarray): Array of shape (t, 1) representing the
        inputs already sampled with the black-box function
            Y_init (numpy.ndarray): Array of shape (t, 1) representing the
        outputs of the black-box function for each input in X_init
            l (float): The length parameter for the kernel
            sigma_f (float): The standard deviation given to the output of the
            kernel
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """Calculates the covariance kernel matrix between two matrices

        Args:
            X1 (numpy.ndarray): Array of shape (m, 1)
            X2 (numpy.ndarray): Array of shape (n, 1)

        Returns:
            numpy.ndarray: The covariance kernel matrix between X1 and X2
        """
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + \
            np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """Predicts the mean and standard deviation of points in a Gaussian
        process

        Args:
            X_s (numpy.ndarray): Array of shape (s, 1) containing all of the
        points whose mean and standard deviation should be calculated

        Returns:
            mu (numpy.ndarray): Array of shape (s,) containing the mean for
        each point in X_s, respectively
            sigma (numpy.ndarray): Array of shape (s,) containing the standard
        deviation for each point in X_s, respectively
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu = K_s.T.dot(K_inv).dot(self.Y).reshape(-1)
        sigma = np.diag(K_ss - K_s.T.dot(K_inv).dot(K_s))
        return mu, sigma
