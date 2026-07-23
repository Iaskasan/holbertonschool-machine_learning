#!/usr/bin/env python3
"""Fills missing values in cryptocurrency market data."""


def fill(df):
    """Remove Weighted_Price and fill missing values by column."""
    df = df.drop(columns=["Weighted_Price"])
    df["Close"] = df["Close"].ffill()

    for column in ["High", "Low", "Open"]:
        df[column] = df[column].fillna(df["Close"])

    volume_columns = ["Volume_(BTC)", "Volume_(Currency)"]
    df[volume_columns] = df[volume_columns].fillna(0)
    return df
