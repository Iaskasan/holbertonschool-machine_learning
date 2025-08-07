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
    if axis == 1:
        return [a+b for a, b in zip(mat1, mat2)]
    if axis == 2:
        new_mat = []
        for i in range(len(mat1)):
            axis2i = []
            for j in range(len(mat1[i])):
                axis2j = []
                for k in range(len(mat1[i][j])):
                    temp = mat1[i][j][k] + mat2[i][j][k]
                    axis2j.append(temp)
                axis2i.append(axis2j)
            new_mat.append(axis2i)
        return new_mat
    if axis == 3:
        new_matrix = []
        for i in range(len(mat1)):
            new_matrix_i = []
            for j in range(len(mat1[i])):
                new_matrix_j = []
                for k in range(len(mat1[i][j])):
                    new_list_k = mat1[i][j][k] + mat2[i][j][k]
                    new_matrix_j.append(new_list_k)
                new_matrix_i += new_matrix_j
            new_matrix += new_matrix_i
        return new_matrix


def mat_shape(matrix):
    """compare the shape of 2 matrices,
    returns None if they dont match"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if len(matrix) == 0:
            break
        matrix = matrix[0]
    return shape
