#!/usr/bin/env python3
""" Concatenates two matrices"""


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenates two matrices along a specific axis"""
    if len(mat1) < 1 or len(mat2) < 1:
        return None
    new_mat = []
    if axis == 0:
        new_mat = mat1 + mat2
        return new_mat
    if axis == 1:
        for i in range(len(mat1)):
            new_mat.append(mat1[i] + mat2[i])
        return new_mat
