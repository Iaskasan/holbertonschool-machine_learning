#!/usr/bin/env python3
"""Squashed like sardines"""


def cat_matrices(mat1, mat2, axis=0):
    """concatenates two matrices along a specific axis"""
    matrix = []
    if len(mat1) != len(mat2):
        return None
    if axis == 0:
        matrix = mat1 + mat2
        return matrix
    if axis == 1:
        matrix = [a+b for a, b in zip(mat1, mat2)]
        return matrix
    if axis == 2:
        matrix = [a+b for a, b in zip(mat1[0], mat2[0])]
        return matrix
    if axis == 3:
        matrix = [a+b for a, b in zip(mat1[0][0], mat2[0][0])]
        return matrix
