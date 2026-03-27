#!/usr/bin/env python3
"""Calculates the determinant of a matrix."""


def determinant(matrix):
    """Calculates the determinant of a matrix."""

    # Check if matrix is a list of lists
    if not isinstance(matrix, list) or \
       not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check for empty matrix [[]] → 0x0
    if matrix == [[]]:
        return 1

    # Check if square
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base case: 1x1
    if n == 1:
        return matrix[0][0]

    # Base case: 2x2
    if n == 2:
        return matrix[0][0] * matrix[1][1] - \
               matrix[0][1] * matrix[1][0]

    # Recursive case
    det = 0
    for j in range(n):
        # Build minor (remove row 0 and column j)
        minor = [
            [matrix[i][k] for k in range(n) if k != j]
            for i in range(1, n)
        ]

        # Cofactor expansion
        det += ((-1) ** j) * matrix[0][j] * determinant(minor)

    return det
