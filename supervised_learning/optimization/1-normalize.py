#!/usr/bin/env python3
"""normalize dataset using provided normalization constants"""
import numpy as np


def normalize(X, m, s):
    """
    Normalize dataset X using mean m and std s.
    """
    X_norm = (X - m) / s
    return X_norm
