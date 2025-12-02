#!/usr/bin/env python3
"""2D matrix transposition"""


def matrix_transpose(matrix):
    "matrix transposition method"
    trans = []
    for i in range(len(matrix[0])):
        trans.append([row[i] for row in matrix])
    return trans
