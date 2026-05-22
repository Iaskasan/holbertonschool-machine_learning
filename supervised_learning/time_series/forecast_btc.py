#!/usr/bin/env python3
"""Train an RNN model to forecast BTC closing price."""

import numpy as np
from pathlib import Path

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "TensorFlow is not installed in this environment. "
        "Install it with: python3 -m pip install tensorflow"
    ) from exc


BATCH_SIZE = 64
EPOCHS = 20
BASE_DIR = Path(__file__).resolve().parent


def make_dataset(X, y, shuffle=False):
    """Create tf.data.Dataset."""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def build_model(input_shape):
    """Build RNN forecasting model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )

    return model


def main():
    """Train and validate BTC forecasting model."""
    data = np.load(BASE_DIR / "btc_preprocessed.npz")

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]

    train_ds = make_dataset(X_train, y_train, shuffle=True)
    val_ds = make_dataset(X_val, y_val)

    model = build_model(X_train.shape[1:])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(BASE_DIR / "btc_forecast_model.keras"),
            monitor="val_loss",
            save_best_only=True
        )
    ]

    model.summary()

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    loss, mae = model.evaluate(val_ds)
    print("Validation MSE:", loss)
    print("Validation MAE:", mae)


if __name__ == "__main__":
    main()
