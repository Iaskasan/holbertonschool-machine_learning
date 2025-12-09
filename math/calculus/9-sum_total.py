#!/usr/bin/env python3
"""sum function"""


def summation_i_squared(n):
    """calculates the summation of i squared
    with n as a stopping condition"""
    if type(n) is not int:
        return None
    return (n**2) * 2 + n
