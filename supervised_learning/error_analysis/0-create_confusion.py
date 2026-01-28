#!/usr/bin/env python3
"""creates a confusion matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix.

    Args:
    labels: np.ndarray of shape (m, classes) - one-hot true labels
    logits: np.ndarray of shape (m, classes) - one-hot predicted labels


    Returns:
    np.ndarray of shape (classes, classes) - confusion matrix
    Rows = true labels, Columns = predicted labels
    """
    true_labels = np.argmax(labels, axis=1)
    pred_labels = np.argmax(logits, axis=1)

    num_classes = labels.shape[1]
    confusion = np.zeros((num_classes, num_classes), dtype=int)

    for t, p in zip(true_labels, pred_labels):
        confusion[t, p] += 1
    return confusion
