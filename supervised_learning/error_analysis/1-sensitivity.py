#!/usr/bin/env python3
"""calculate the sensitivity for each class in a confusion matrix"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity (recall or true positive rate) for each class
    in a confusion matrix.

    Sensitivity measures the proportion of actual positives that are correctly
    identified by the classifier.

    Args:
        confusion (numpy.ndarray): A square confusion matrix of shape
            (classes, classes), where rows represent the true labels and
            columns represent the predicted labels.

    Returns:
        numpy.ndarray: A 1D array of shape (classes,) containing the
        sensitivity for each class, where:
            sensitivity[i] = TP_i / (TP_i + FN_i)

        - TP_i (True Positives): Number of samples correctly classified
        as class i
          (diagonal element confusion[i, i]).
        - FN_i (False Negatives): Number of samples of class i that were
          misclassified (sum of row i excluding the diagonal).
    """
    TP = np.diag(confusion)
    FN = confusion.sum(axis=1) - TP
    return TP / (TP + FN)
