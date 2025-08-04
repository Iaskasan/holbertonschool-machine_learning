#!/usr/bin/env python3
"""yes life"""


def add_matrices2D(mat1, mat2):
    """adds 2 2D matrices (r1 and r2 means row1 and row2
    but it was too long for pycodestyle)"""
    if len(mat1[0]) != len(mat2[0]):
        return None
    return [[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(mat1, mat2)]
