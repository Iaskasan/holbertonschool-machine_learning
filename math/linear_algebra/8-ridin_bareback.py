#!/usr/bin/env python3
"""Ridin' Bareback"""


def mat_mul(mat1, mat2):
    """Performs matrix multiplication"""
    if not mat1 or not mat2:
        return None
    if len(mat1[0]) != len(mat2):
        return None
    new_matrix = []
    for row in mat1:
        new_row = []
        for j in range(len(mat2[0])):
            dot_product = sum(row[k] * mat2[k][j] for k in range(len(mat2)))
            new_row.append(dot_product)
        new_matrix.append(new_row)
    return new_matrix
