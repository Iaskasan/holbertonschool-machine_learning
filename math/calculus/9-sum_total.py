#!/usr/bin/env python3
"""sum function"""


def summation_i_squared(n):
    """calculates the summation of i squared
    with n as a stopping condition"""
    sum = 0
    for i in range(1, n + 1):
        sum += i**2
    return sum
