#!/usr/bin/env python3
"""Converts selected DataFrame values into a NumPy array."""


def array(df):
    """Return the last 10 rows of the High and Close columns as an array."""
    return df[["High", "Close"]].tail(10).to_numpy()
