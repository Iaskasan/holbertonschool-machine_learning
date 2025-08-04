#!/usr/bin/env python3
"""Getting cozy"""


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenate 2 2D matrices along a specific axis"""
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return list(mat1 + mat2)
    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [[*row1, *row2] for row1, row2 in zip(mat1, mat2)]
