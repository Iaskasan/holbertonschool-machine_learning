#!/usr/bin/env python3
"""Converts a numpy.ndarray into a pd.DataFrame"""
import pandas as pd


def from_numpy(array):
    """Converts a numpy.ndarray into a pd.DataFrame"""
    return pd.DataFrame(
        array, columns=[chr(65 + i) for i in range(array.shape[1])])
