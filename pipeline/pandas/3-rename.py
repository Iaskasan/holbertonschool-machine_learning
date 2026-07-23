#!/usr/bin/env python3
"""Renames and converts a timestamp column in a DataFrame."""
import pandas as pd


def rename(df):
    """Rename Timestamp, convert it to datetime, and keep selected columns."""
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
    return df[["Datetime", "Close"]]
