#!/usr/bin/env python3
"""calculates F1 score for a confusion matrix"""
import numpy as np


def f1_score(confusion):
    """
    Calculates F1 score for each class in a confusion matrix.

    Args:
    confusion: np.ndarray of shape (classes, classes) - confusion matrix
               Rows = true labels, Columns = predicted labels

    Returns:
    np.ndarray of shape (classes,) - F1 score for each class
    """
    true_positives = np.diag(confusion)
    false_positives = np.sum(confusion, axis=0) - true_positives
    false_negatives = np.sum(confusion, axis=1) - true_positives

    with np.errstate(divide='ignore', invalid='ignore'):
        precision_values = true_positives / (
            true_positives + false_positives)
        precision_values = np.nan_to_num(precision_values)

        sensitivity_values = true_positives / (
            true_positives + false_negatives)
        sensitivity_values = np.nan_to_num(sensitivity_values)

        f1_scores = 2 * (precision_values * sensitivity_values) / (
            precision_values + sensitivity_values)
        f1_scores = np.nan_to_num(f1_scores)

    return f1_scores
