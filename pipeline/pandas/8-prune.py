#!/usr/bin/env python3
"""Removes any entries where Close has NaN values."""


def prune(df):
    """Return the DataFrame without rows containing NaN in Close."""
    return df.dropna(subset=["Close"])
