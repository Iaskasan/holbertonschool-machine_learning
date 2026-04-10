#!/usr/bin/env python3
"""Performs the expectation maximization for a GMM."""
import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation maximization for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        k: positive integer containing the number of clusters
        iterations: positive integer containing the maximum number of
                    iterations for the algorithm
        tol: non-negative float containing tolerance of the log likelihood
        verbose: boolean that determines if information should be printed

    Returns:
        pi, m, S, g, l
        or None, None, None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    prev_l = None

    for i in range(iterations + 1):
        g, log = expectation(X, pi, m, S)
        if g is None or log is None:
            return None, None, None, None, None

        if verbose and (i % 10 == 0 or i == iterations):
            print("Log Likelihood after {} iterations: {:.5f}".format(i, log))

        if prev_l is not None and abs(log - prev_l) <= tol:
            return pi, m, S, g, log

        if i == iterations:
            return pi, m, S, g, log

        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        prev_l = log
