#!/usr/bin/env python3
"""Concatenates market data using a chronological MultiIndex."""
import pandas as pd


index = __import__('10-index').index


def hierarchy(df1, df2):
    """Combine coinbase and bitstamp data within the timestamp interval."""
    df1 = index(df1)
    df2 = index(df2)

    start = 1417411980
    end = 1417417980
    df1 = df1[(df1.index >= start) & (df1.index <= end)]
    df2 = df2[(df2.index >= start) & (df2.index <= end)]

    combined = pd.concat([df2, df1], keys=["bitstamp", "coinbase"])
    return combined.swaplevel(0, 1).sort_index()
