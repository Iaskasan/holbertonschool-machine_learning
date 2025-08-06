#!/usr/bin/env python3
"""Squashed like sardines"""


def cat_matrices(mat1, mat2, axis=0):
    """concatenates two matrices along a specific axis"""
    if len(mat1) != len(mat2):
        return None
    if axis == 0:
        return mat1 + mat2
    if axis == 1:
        return [a+b for a, b in zip(mat1, mat2)]
    if axis == 2:
        print("axis 2:")
        return [a+b for a, b in zip(mat1[0], mat2[0])]
    if axis == 3:
        print("axis 3:")
        return [a+b for a, b in zip(mat1[0][0], mat2[0][0])]
