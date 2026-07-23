#!/usr/bin/env python3
"""Sorts a Dataframe by price in descending order"""


def high(df):
    """Sorts a Dataframe by price in descending order"""
    return df.sort_values(by="High", ascending=False)
