#!/usr/bin/env python3
"""Calculates the PDF of a Gaussian distribution."""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the PDF of a Gaussian distribution.

    Args:
        X: numpy.ndarray of shape (n, d)
        m: numpy.ndarray of shape (d,)
        S: numpy.ndarray of shape (d, d)

    Returns:
        P: numpy.ndarray of shape (n,)
        or None on failure
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
        not isinstance(m, np.ndarray) or m.ndim != 1 or
        not isinstance(S, np.ndarray) or S.ndim != 2):
        return None

    n, d = X.shape

    if m.shape != (d,) or S.shape != (d, d):
        return None

    try:
        det = np.linalg.det(S)
        inv = np.linalg.inv(S)
    except Exception:
        return None

    if det <= 0:
        return None

    # Center data
    X_centered = X - m  # shape (n, d)

    # Compute exponent: (x - m)^T S^-1 (x - m)
    exponent = np.sum((X_centered @ inv) * X_centered, axis=1)

    # Compute normalization constant
    norm_const = 1 / np.sqrt((2 * np.pi) ** d * det)

    # Compute PDF
    P = norm_const * np.exp(-0.5 * exponent)

    # Enforce minimum value
    P = np.maximum(P, 1e-300)

    return P
