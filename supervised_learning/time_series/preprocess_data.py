#!/usr/bin/env python3
"""Preprocess BTC data for hourly forecasting."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path


WINDOW = 24
TARGET_COL = "Close"
FEATURES = ["Open", "High", "Low", "Close", "Volume_(BTC)",
            "Volume_(Currency)", "Weighted_Price"]
BASE_DIR = Path(__file__).resolve().parent


def load_data(path):
    """Load one BTC dataset."""
    df = pd.read_csv(path)

    df.columns = [
        "Timestamp", "Open", "High", "Low", "Close",
        "Volume_(BTC)", "Volume_(Currency)", "Weighted_Price"
    ]

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")
    df = df.set_index("Timestamp")

    return df


def main():
    """Preprocess Coinbase and Bitstamp BTC datasets."""
    coinbase = load_data(BASE_DIR / "coinbase.csv")
    bitstamp = load_data(BASE_DIR / "bitstamp.csv")

    df = pd.concat([coinbase, bitstamp])
    df = df.sort_index()

    df = df[FEATURES]
    df = df.dropna()

    # Convert minute data into hourly data
    hourly = df.resample("1h").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume_(BTC)": "sum",
        "Volume_(Currency)": "sum",
        "Weighted_Price": "mean"
    })

    hourly = hourly.dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(hourly)

    X = []
    y = []

    close_index = FEATURES.index(TARGET_COL)

    for i in range(WINDOW, len(scaled) - 1):
        X.append(scaled[i - WINDOW:i])
        y.append(scaled[i + 1, close_index])

    X = np.array(X)
    y = np.array(y)

    train_size = int(len(X) * 0.8)

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:]
    y_val = y[train_size:]

    np.savez_compressed(
        BASE_DIR / "btc_preprocessed.npz",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val
    )

    joblib.dump(scaler, BASE_DIR / "btc_scaler.save")

    print("Preprocessing complete.")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)


if __name__ == "__main__":
    main()
