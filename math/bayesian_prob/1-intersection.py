#!/usr/bin/env python3
"""Calculates the intersection of obtaining the data
with various hypothetical probabilities."""
import numpy as np


def likelihood(x, n, P):
    """Calculates likelihood using binomial distribution"""

    # Validate n
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    # Validate x
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    # Validate P
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Compute binomial coefficient (no math import)
    coeff = 1
    for i in range(1, x + 1):
        coeff *= (n - x + i) / i

    # Likelihood
    return coeff * (P ** x) * ((1 - P) ** (n - x))


def intersection(x, n, P, Pr):
    """Calculates the intersection"""

    # Validate n
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    # Validate x
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    # Validate P
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    # Validate Pr
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    # Check P values
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Check Pr values
    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    # Check sum of Pr
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    return likelihood(x, n, P) * Pr
