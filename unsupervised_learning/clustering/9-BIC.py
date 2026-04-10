#!/usr/bin/env python3
"""Finds the best number of clusters for a GMM using BIC."""
import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using BIC."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None
    if kmax is None:
        kmax = X.shape[0]
    if not isinstance(kmax, int) or kmax <= 0:
        return None, None, None, None
    if kmin > kmax:
        return None, None, None, None
    if kmax > X.shape[0]:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape
    ks = np.arange(kmin, kmax + 1)
    l = np.zeros(ks.shape)
    b = np.zeros(ks.shape)
    results = []

    for i, k in enumerate(ks):
        k = int(k)
        pi, m, S, _, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose
        )
        if pi is None:
            return None, None, None, None

        p = (k - 1) + (k * d) + (k * d * (d + 1) / 2)
        l[i] = log_likelihood
        b[i] = p * np.log(n) - 2 * log_likelihood
        results.append((pi, m, S))

    best = np.argmin(b)
    best_k = int(ks[best])
    best_result = results[best]

    return best_k, best_result, l, b
