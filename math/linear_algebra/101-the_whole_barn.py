#!/usr/bin/env python3
"""The whole barn"""


def add_matrices(mat1, mat2):
    """adds two matrices"""
    array1 = np.array(mat1)
    array2 = np.array(mat2)
    if array1.shape != array2.shape:
        return None
    return (array1 + array2).tolist()
