#!/usr/bin/env python3
"""create a confusion matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix.

    Args:
        labels: one-hot numpy.ndarray of shape (m, classes),
                true labels
        logits: one-hot numpy.ndarray of shape (m, classes),
                predicted labels

    Returns:
        confusion: numpy.ndarray of shape (classes, classes),
                   where rows = actual labels,
                   columns = predicted labels
    """
    true_classes = np.argmax(labels, axis=1)
    pred_classes = np.argmax(logits, axis=1)
    classes = labels.shape[1]
    confusion = np.zeros((classes, classes), dtype=float)
    for t, p in zip(true_classes, pred_classes):
        confusion[t, p] += 1
    return confusion
