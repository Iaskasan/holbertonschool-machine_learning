#!/usr/bin/env python3
"""Adds two arrays element-wise"""
import numpy as np


def add_arrays(arr1, arr2):
    """Adds two arrays element-wise."""
    if len(arr1) != len(arr2):
        return None
    return list(map(int, np.add(arr1, arr2)))
