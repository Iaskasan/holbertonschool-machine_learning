#!/usr/bin/env python3
"""Squashed like sardines"""


def cat_matrices(mat1, mat2, axis=0):
    """concatenates two matrices along a specific axis"""
    shape1 = mat_shape(mat1)
    shape2 = mat_shape(mat2)
    if shape1 is None or shape2 is None:
        return None
    if len(shape1) != len(shape2):
        return None

    if axis == 0:
        return mat1 + mat2
    if axis > 0:
        if len(mat1) != len(mat2):
            return None
        return [(cat_matrices(sub1, sub2, axis - 1))
                for sub1, sub2 in zip(mat1, mat2)]


def mat_shape(matrix):
    """gets the shape a matrix"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if len(matrix) == 0:
            break
        matrix = matrix[0]
    return shape
