#!/usr/bin/env python3
"""Getting cozy"""


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenate 2 2D matrices along a specific axis"""
    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return None
    if axis == 0:
        return list(mat1 + mat2)
    if axis == 1:
        return [[*row1, *row2] for row1, row2 in zip(mat1, mat2)]
