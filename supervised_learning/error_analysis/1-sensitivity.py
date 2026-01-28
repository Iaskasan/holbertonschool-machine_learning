#!/usr/bin/env python3
"""calculates sensitivity for each class in a confusion matrix"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates sensitivity for each class in a confusion matrix.

    Args:
    confusion: np.ndarray of shape (classes, classes) - confusion matrix
               Rows = true labels, Columns = predicted labels

    Returns:
    np.ndarray of shape (classes,) - sensitivity for each class
    """
    true_positives = np.diag(confusion)
    false_negatives = np.sum(confusion, axis=1) - true_positives

    with np.errstate(divide='ignore', invalid='ignore'):
        sensitivity_values = true_positives / (
            true_positives + false_negatives)
        sensitivity_values = np.nan_to_num(sensitivity_values)

    return sensitivity_values
