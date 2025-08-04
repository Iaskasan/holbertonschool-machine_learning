#!/usr/bin/env python3

def matrix_transpose(matrix):
    """Returns the transpose of a matrix."""
    transpose = list(zip(*matrix))
    transpose_list = list(map(list, transpose))
    return transpose_list
