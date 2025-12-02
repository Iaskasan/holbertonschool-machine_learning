#!/usr/bin/env python3
"""Adding 2 arrays"""


def add_arrays(arr1, arr2):
    """adds two arrays element-wise"""
    sum = []
    if len(arr1) != len(arr2):
        return None
    for i in range(len(arr1)):
        sum.append(arr1[i] + arr2[i])
    return sum
