#!/usr/bin/env python3
"""calculates specificity for each class in a confusion matrix"""
import numpy as np


def specificity(confusion):
    """
    Calculates specificity for each class in a confusion matrix.

    Args:
    confusion: np.ndarray of shape (classes, classes) - confusion matrix
               Rows = true labels, Columns = predicted labels

    Returns:
    np.ndarray of shape (classes,) - specificity for each class
    """
    true_negatives = np.sum(confusion) - (
        np.sum(confusion, axis=1) + np.sum(
            confusion, axis=0) - np.diag(confusion))
    false_positives = np.sum(confusion, axis=0) - np.diag(confusion)

    with np.errstate(divide='ignore', invalid='ignore'):
        specificity_values = true_negatives / (
            true_negatives + false_positives)
        specificity_values = np.nan_to_num(specificity_values)

    return specificity_values
