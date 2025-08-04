#!/usr/bin/env python3
"""Returns the shape of a matrix as a list of dimensions."""


def matrix_shape(matrix):
    """Returns the shape of a matrix as a list of dimensions."""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
