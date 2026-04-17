#!/usr/bin/env python3
"""Bayesian optimization with Gaussian Processes"""
from scipy.stats import norm
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """Performs Bayesian optimization on a noiseless 1D Gaussian process"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """Class constructor

        Args:
            f (function): The black-box function to be optimized
            X_init (numpy.ndarray): Array of shape (t, 1) representing the
        inputs already sampled with the black-box function
            Y_init (numpy.ndarray): Array of shape (t, 1) representing the
        outputs of the black-box function for each input in X_init
            bounds (tuple): Tuple of (min, max) representing the bounds of the
        space in which to look for the optimal point
            ac_samples (int): The number of samples that should be analyzed
        during acquisition
            l (float): The length parameter for the kernel
            sigma_f (float): The standard deviation given to the output of the
            kernel
            xsi (float): The exploration-exploitation factor for acquisition
            minimize (bool): A bool determining whether optimization should be
        performed for minimization (True) or maximization (False)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.xsi = xsi
        self.minimize = minimize
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)

    def acquisition(self):
        """Calculates the next best sample location

        Returns:
            X_next (numpy.ndarray): Array of shape (1,) representing the next
        best sample point
        """
        mu, sigma = self.gp.predict(self.X_s)
        if self.minimize:
            Y_opt = np.min(self.gp.Y)
            imp = Y_opt - mu - self.xsi
        else:
            Y_opt = np.max(self.gp.Y)
            imp = mu - Y_opt - self.xsi
        Z = imp / sigma
        EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        X_next = self.X_s[np.argmax(EI)]
        return X_next.reshape(1,), EI

    def optimize(self, iterations=100):
        """Optimizes the black-box function

        Args:
            iterations (int): The maximum number of iterations to perform

        Returns:
            X_opt (numpy.ndarray): Array of shape (1,) representing the optimal
        point
            Y_opt (numpy.ndarray): Array of shape (1,) representing the optimal
        function value
        """
        for i in range(iterations):
            X_next, EI = self.acquisition()
            if EI == 0:
                break
            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)
        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)
        X_opt = self.gp.X[idx]
        Y_opt = self.gp.Y[idx]
        return X_opt, Y_opt
