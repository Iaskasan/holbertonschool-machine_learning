#!/usr/bin/env python3
"""calculates the precision for each class in a confusion matrix"""
import numpy as np


def precision(confusion):
    """
    Calculates precision for each class in a confusion matrix.

    Args:
    confusion: np.ndarray of shape (classes, classes) - confusion matrix
               Rows = true labels, Columns = predicted labels

    Returns:
    np.ndarray of shape (classes,) - precision for each class
    """
    true_positives = np.diag(confusion)
    false_positives = np.sum(confusion, axis=0) - true_positives
    with np.errstate(divide='ignore', invalid='ignore'):
        precision_values = true_positives / (
            true_positives + false_positives)
        precision_values = np.nan_to_num(precision_values)

    return precision_values
