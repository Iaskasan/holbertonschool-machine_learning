#!/usr/bin/env python3
"""Class that represents a Multivariate Normal distribution."""
import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution."""

    def __init__(self, data):
        """Initialize the distribution from a dataset."""
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        centered = data - self.mean
        self.cov = (centered @ centered.T) / (n - 1)

    def pdf(self, x):
        """Calculates the PDF at a data point."""

        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]

        if x.shape != (d, 1):
            raise ValueError(f"x must have the shape ({d}, 1)")

        diff = x - self.mean

        # determinant and inverse of covariance
        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)

        # normalization constant
        norm_const = 1 / np.sqrt(((2 * np.pi) ** d) * det)

        # exponent
        exponent = -0.5 * (diff.T @ inv @ diff)

        return float(norm_const * np.exp(exponent))
