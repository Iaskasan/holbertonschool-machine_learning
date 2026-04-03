#!/usr/bin/env python3
"""Performs PCA on a dataset."""
import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset and returns the transformed data.

    Parameters:
    X (np.ndarray): shape (n, d) dataset
    ndim (int): target number of dimensions

    Returns:
    np.ndarray: shape (n, ndim) transformed data
    """
    # Step 1: Center the data
    X_centered = X - np.mean(X, axis=0)

    # Step 2: Covariance matrix
    cov = np.cov(X_centered, rowvar=False)

    # Step 3: Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Step 4: Sort eigenvectors by descending eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Step 5: Select top ndim components
    W = eigenvectors[:, :ndim]

    # Step 6: Project data
    T = X_centered @ W

    return T
