#!/usr/bin/env python3
"""Calculates the maximization step in the EM algorithm for a GMM."""
import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        g: numpy.ndarray of shape (k, n) containing the posterior
           probabilities for each data point in each cluster

    Returns:
        pi, m, S
        pi is a numpy.ndarray of shape (k,) containing the updated priors
        m is a numpy.ndarray of shape (k, d) containing the updated means
        S is a numpy.ndarray of shape (k, d, d) containing the updated
        covariance matrices

        Returns (None, None, None) on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape

    if n_g != n:
        return None, None, None

    if not np.isclose(np.sum(g, axis=0), np.ones(n)).all():
        return None, None, None

    nk = np.sum(g, axis=1)
    if np.any(nk == 0):
        return None, None, None

    pi = nk / n
    m = (g @ X) / nk[:, np.newaxis]

    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]
        weighted_diff = g[i][:, np.newaxis] * diff
        S[i] = (weighted_diff.T @ diff) / nk[i]

    return pi, m, S
