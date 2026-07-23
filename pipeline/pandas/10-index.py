#!/usr/bin/env python3
"""Sets the Timestamp col as the index of the dataframe"""


def index(df):
    return df.set_index("Timestamp")
