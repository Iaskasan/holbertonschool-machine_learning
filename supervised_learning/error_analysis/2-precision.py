#!/usr/bin/env python3
"""calculates the precision for each class in a confusion matrix"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.

    Precision (also called Positive Predictive Value) measures how many of the
    samples predicted as a given class are actually of that class.

    Args:
        confusion (numpy.ndarray): A square confusion matrix of shape
            (classes, classes), where rows represent the true labels and
            columns represent the predicted labels.

    Returns:
        numpy.ndarray: A 1D array of shape (classes,) containing the precision
        for each class, where:
            precision[i] = TP_i / (TP_i + FP_i)

        - TP_i (True Positives): Number of samples correctly predicted
        as class i
          (diagonal element confusion[i, i]).
        - FP_i (False Positives): Number of samples predicted as class i but
          belonging to another class (column sum of i excluding the diagonal).
    """
    TP = np.diag(confusion)
    FP = confusion.sum(axis=0) - TP
    return TP / (FP + TP)
