#!/usr/bin/env python3
"""adds two matrices"""


def add_matrices2D(mat1, mat2):
    """adds two matrices elements-wise"""
    if len(mat1[0]) != len(mat2[0]):
        return None
    sum = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat1[0])):
            row.append(mat1[i][j] + mat2[i][j])
        sum.append(row)
    return sum
