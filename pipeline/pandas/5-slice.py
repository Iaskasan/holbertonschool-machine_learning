#!/usr/bin/env python3
"""Extracts a slice of a DataFrame."""


def slice(df):
    """Extracts a slice of a DataFrame."""
    return df[["High", "Low", "Close", "Volume_(BTC)"]].iloc[::60, :]
