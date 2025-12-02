#!/usr/bin/env python3
"""Returns the shape of a matrix"""


def matrix_shape(matrix):
    """gets the shape of a given matrix of any dimension"""
    size = []
    if type(matrix) is list:
        size.append(len(matrix))
        size += matrix_shape(matrix[0])
    return size
