#!/usr/bin/env python3
"""calculates F1 score for a confusion matrix"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates F1 score for each class in a confusion matrix.

    Args:
    confusion: np.ndarray of shape (classes, classes) - confusion matrix
               Rows = true labels, Columns = predicted labels

    Returns:
    np.ndarray of shape (classes,) - F1 score for each class
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * precision(confusion) * sensitivity(confusion) / (
            precision(confusion) + sensitivity(confusion))
        f1_scores = np.nan_to_num(f1_scores)

    return f1_scores
