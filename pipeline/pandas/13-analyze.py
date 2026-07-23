#!/usr/bin/env python3
"""Computes descriptive statistics of cryptocurrency data."""


def analyze(df):
    """Computes descriptive statistics of cryptocurrency data."""
    df_dropped = df.drop(columns=["Timestamp"])
    return df_dropped.describe()
