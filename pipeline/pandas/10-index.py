#!/usr/bin/env python3
"""Sets the Timestamp col as the index of the dataframe"""


def index(df):
    """Sets the Timestamp col as the index of the dataframe"""
    return df.set_index("Timestamp")
