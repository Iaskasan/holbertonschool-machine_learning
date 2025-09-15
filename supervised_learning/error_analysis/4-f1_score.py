#!/usr/bin/env python3
"""Calculates the F1 score for each class in a confusion matrix"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score for each class in a confusion matrix.

    The F1 score is the harmonic mean of precision and sensitivity (recall),
    and provides a balance between the two metrics. It is useful when you
    want to consider both false positives and false negatives.

    For each class i, the F1 score is defined as:

    F1_i = 2 * (precision_i * sensitivity_i) / (precision_i + sensitivity_i)

    If both precision and sensitivity are 0 for a class, the F1 score is
    defined as 0 to avoid division by zero.

    Args:
        confusion (numpy.ndarray): A square confusion matrix of shape
            (classes, classes), where rows represent the true labels and
            columns represent the predicted labels.

    Returns:
        numpy.ndarray: A 1D array of shape (classes,) containing the F1 score
        for each class, with values in the range [0, 1].
    """
    numerator = precision(confusion) * sensitivity(confusion)
    denominator = precision(confusion) + sensitivity(confusion)
    result = 2 * (numerator / denominator)
    return np.where(denominator == 0, 0, result)
