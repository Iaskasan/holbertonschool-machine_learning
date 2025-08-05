#!/usr/bin/env python3
"""The whole barn"""


def add_matrices(mat1, mat2):
    """Adds two matrices of the same shape (lists of lists) without NumPy."""
    if isinstance(mat1, list):
        if len(mat1) != len(mat2):
            return None
        result = []
        for a, b in zip(mat1, mat2):
            sub_result = add_matrices(a, b)
            if sub_result is None:
                return None
            result.append(sub_result)
        return result
    else:
        return mat1 + mat2
