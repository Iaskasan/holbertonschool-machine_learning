#!/usr/bin/env python3
"""Slice like a ninja"""


def np_slice(matrix, axes={}):
    """slices a matrix along specific axes"""
    slices = [slice(None)] * matrix.ndim
    for axis, slice_tuple in axes.items():
        slices[axis] = slice(*slice_tuple)
    return matrix[tuple(slices)]
