#!/usr/bin/env python3
"""Calculates the minor matrix of a matrix."""


def determinant(matrix):
    """Helper function to compute determinant."""
    if matrix == [[]]:
        return 1

    n = len(matrix)

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - \
               matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(n):
        minor = [
            [matrix[i][k] for k in range(n) if k != j]
            for i in range(1, n)
        ]
        det += ((-1) ** j) * matrix[0][j] * determinant(minor)

    return det


def minor(matrix):
    """Calculates the minor matrix of a matrix."""

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

    # Build minor matrix
    minor_matrix = []

    for i in range(n):
        row_minors = []
        for j in range(n):
            # Build submatrix excluding row i and column j
            submatrix = [
                [matrix[r][c] for c in range(n) if c != j]
                for r in range(n) if r != i
            ]

            row_minors.append(determinant(submatrix))

        minor_matrix.append(row_minors)

    return minor_matrix
