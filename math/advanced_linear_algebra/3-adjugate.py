#!/usr/bin/env python3
"""Calculates the adjugate matrix of a matrix."""


def determinant(matrix):
    """Calculate the determinant of a square matrix."""
    if matrix == [[]]:
        return 1

    n = len(matrix)

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return (matrix[0][0] * matrix[1][1] -
                matrix[0][1] * matrix[1][0])

    det = 0
    for j in range(n):
        submatrix = [
            [matrix[i][k] for k in range(n) if k != j]
            for i in range(1, n)
        ]
        det += ((-1) ** j) * matrix[0][j] * determinant(submatrix)

    return det


def cofactor(matrix):
    """Calculate the cofactor matrix."""
    n = len(matrix)

    if n == 1:
        return [[1]]

    cof = []
    for i in range(n):
        row = []
        for j in range(n):
            submatrix = [
                [matrix[r][c] for c in range(n) if c != j]
                for r in range(n) if r != i
            ]
            minor = determinant(submatrix)
            row.append(((-1) ** (i + j)) * minor)
        cof.append(row)

    return cof


def adjugate(matrix):
    """Calculates the adjugate matrix of a matrix."""

    # Type check
    if not isinstance(matrix, list) or \
       not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check non-empty square
    if matrix == [] or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Special case: 1x1
    if n == 1:
        return [[1]]

    # Get cofactor matrix
    cof = cofactor(matrix)

    # Transpose to get adjugate
    adj = [[cof[j][i] for j in range(n)] for i in range(n)]

    return adj
