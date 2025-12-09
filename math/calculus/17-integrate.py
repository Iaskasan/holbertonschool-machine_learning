#!/usr/bin/env python3
"""calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """calculates the integral of a polynomial"""
    if not isinstance(poly, list) or not isinstance(C, int):
        return None
    if any(not isinstance(c, (int, float)) for c in poly):
        return None
    if len(poly) < 1:
        return None

    result = [C]
    for i in range(len(poly)):
        val = poly[i] / (i + 1)
        if isinstance(val, float) and val.is_integer():
            val = int(val)
        result.append(val)
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result
