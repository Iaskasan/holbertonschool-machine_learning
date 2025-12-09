#!/usr/bin/env python3
"""sum function"""


def summation_i_squared(n):
    """calculates the summation of i squared
    with n as a stopping condition"""
    if type(n) is not int or n < 1:
        return None
    return ((n * (n + 1)) * (2 * n + 1)) / 6
