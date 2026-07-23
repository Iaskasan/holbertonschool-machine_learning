#!/usr/bin/env python3
"""Cleans, aggregates, and visualizes cryptocurrency market data."""
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')


def visualize(df):
    """Transform market data into daily values, plot it, and return it."""
    df = df.drop(columns=["Weighted_Price"])
    df = df.rename(columns={"Timestamp": "Date"})
    df["Date"] = pd.to_datetime(df["Date"], unit="s")
    df = df.set_index("Date")

    df["Close"] = df["Close"].ffill()
    for column in ["High", "Low", "Open"]:
        df[column] = df[column].fillna(df["Close"])

    volume_columns = ["Volume_(BTC)", "Volume_(Currency)"]
    df[volume_columns] = df[volume_columns].fillna(0)

    df = df.loc["2017":]
    df = df.resample("D").agg({
        "High": "max",
        "Low": "min",
        "Open": "mean",
        "Close": "mean",
        "Volume_(BTC)": "sum",
        "Volume_(Currency)": "sum",
    })
    print(df)
    df.plot()
    plt.show()
    return df


if __name__ == "__main__":
    visualize(df)
