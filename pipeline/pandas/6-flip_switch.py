#!/usr/bin/env python3
"""sorts the data in reverse chronological order and transposes it"""


def flip_switch(df):
    """sorts the data in reverse chronological order and transposes it"""
    df = df.sort_values(by="Timestamp", ascending=False)
    return df.T
