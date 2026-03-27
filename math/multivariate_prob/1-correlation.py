#!/usr/bin/env python3
"""Calculates the correlation matrix from a covariance matrix."""
import numpy as np


def correlation(C):
    """Calculates the correlation matrix from a covariance matrix."""

    # Check type
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    # Check shape
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    # Standard deviations (sqrt of diagonal)
    std = np.sqrt(np.diag(C))

    # Outer product of std deviations
    denom = np.outer(std, std)

    # Avoid division by zero (optional but safe)
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = C / denom
        corr[denom == 0] = 0

    return corr
