#!/usr/bin/env python3
"""Calculates the specificity (true negative rate) for each class
    in a confusion matrix"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity (true negative rate) for each class
    in a confusion matrix.

    Specificity measures the proportion of actual negatives that are
    correctly identified by the classifier. For class i, it is defined as:

        specificity[i] = TN_i / (TN_i + FP_i)

    where:
        - TN_i (True Negatives): Number of samples that are not in class i
          and were also not predicted as class i.
        - FP_i (False Positives): Number of samples that are not in class i
          but were incorrectly predicted as class i.

    Args:
        confusion (numpy.ndarray): A square confusion matrix of shape
            (classes, classes), where rows represent the true labels and
            columns represent the predicted labels.

    Returns:
        numpy.ndarray: A 1D array of shape (classes,) containing the
        specificity for each class, with values in the range [0, 1].
    """
    TP = np.diag(confusion)
    FP = confusion.sum(axis=0) - TP
    FN = confusion.sum(axis=1) - TP
    ALL = confusion.sum()
    TN = ALL - (TP + FP + FN)
    return TN / (TN + FP)
