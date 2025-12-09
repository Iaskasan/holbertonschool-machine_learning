#!/usr/bin/env python3
"""Calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """Outputs a list of the derivatives
    of a given polynomial list"""
    if type(poly) is not list or len(poly) < 1:
        return None
    result = []
    for i in range(len(poly)):
        if i == 0:
            continue
        result.append(poly[i] * i)
    if len(result) == 0 or all(x == 0 for x in result):
        return [0]
    return result
