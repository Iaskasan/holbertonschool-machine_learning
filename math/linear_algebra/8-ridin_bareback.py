#!/usr/bin/env python3
"""performs a matrix multiplication"""


def mat_mul(mat1, mat2):
    """performs matrix multiplication of two 2D matrices"""
    if len(mat1) == 0 or len(mat2) == 0:
        return None
    if len(mat1[0]) != len(mat2):
        return None
    result = []
    for i in range(len(mat1)):
        new_row = []
        for j in range(len(mat2[0])):
            element = 0
            for k in range(len(mat2)):
                element += mat1[i][k] * mat2[k][j]
            new_row.append(element)
        result.append(new_row)
    return result
