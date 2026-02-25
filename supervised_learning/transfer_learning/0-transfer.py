#!/usr/bin/env python3
"""
Fine-tune EfficientNetV2S on CIFAR-10 using transfer learning.
"""

import csv
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, mixed_precision, models, regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical


# Tweaking section, here you can modify saving directories, verbose,
# fine-tuning parameters...
MODEL_NAME = "efficientnetv2s_20ep"
BASE_MODEL = tf.keras.applications.EfficientNetV2S
TARGET_SIZE = (224, 224)

EPOCHS_STAGE1 = 20
EPOCHS_STAGE2 = 15
BATCH_SIZE = 128
EARLY_STOPPING = 4
VALIDATION_SPLIT = 0.1
SEED = 42

LR_STAGE1 = 1e-4
LR_STAGE2 = 1e-6
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1

UNFREEZE_LAYERS = 40
DROPOUT1 = 0.5
DROPOUT2 = 0.3
DENSE1 = 512
DENSE2 = 256

# EfficientNetV2S performs best with built-in preprocessing enabled.
PREPROCESSING = True
USE_MIXED_PRECISION = True

SAVE_DIR = f"trained_models/{MODEL_NAME}"
PLOTS_DIR = f"training_plots/{MODEL_NAME}"
LOGS_DIR = "logs"
LOG_FILE = "training_results.csv"

ENABLE_PLOTTING = True
SAVE_PLOTS = True
SHOW_PLOTS = False


if USE_MIXED_PRECISION:
    mixed_precision.set_global_policy("mixed_float16")


def preprocess_data(X, Y):
    X = X.astype("float32")
    Y = to_categorical(Y, 10)
    return X, Y


def split_train_validation(X, Y, val_fraction=VALIDATION_SPLIT, seed=SEED):
    """Split training data into train/validation once, reproducibly."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)

    val_size = int(len(X) * val_fraction)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    return X[train_idx], Y[train_idx], X[val_idx], Y[val_idx]


def build_augmentation():
    """Data augmentation for CIFAR-10 transferred to larger input size."""
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.15),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )


def build_model(input_shape):
    """Builds a transfer-learning model using EfficientNetV2S as base."""
    inputs = layers.Input(shape=input_shape)

    x = build_augmentation()(inputs)
    x = layers.Resizing(*TARGET_SIZE)(x)

    base_model = BASE_MODEL(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
        include_preprocessing=PREPROCESSING,
    )
    base_model.trainable = False

    y = layers.GlobalAveragePooling2D()(base_model.output)
    y = layers.BatchNormalization()(y)
    y = layers.Dense(
        DENSE1,
        activation="relu",
        kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
    )(y)
    y = layers.Dropout(DROPOUT1)(y)
    y = layers.Dense(
        DENSE2,
        activation="relu",
        kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
    )(y)
    y = layers.Dropout(DROPOUT2)(y)

    # Keep final layer in float32 for numeric stability with mixed precision.
    outputs = layers.Dense(10, activation="softmax", dtype="float32")(y)

    model = models.Model(inputs, outputs, name=f"cifar10_{MODEL_NAME}")
    return model, base_model


def compile_and_train(model, lr, epochs, X_train, Y_train, X_val, Y_val, phase):
    """Compile and train with given hyperparameters."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=LABEL_SMOOTHING,
        ),
        metrics=["accuracy"],
    )

    checkpoint_path = os.path.join(
        SAVE_DIR,
        f"{model.name}_{phase.lower().replace(' ', '_').replace('(', '').replace(')', '')}_best.keras",
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=EARLY_STOPPING,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
    ]

    print(f"\nStarting {phase} training ({epochs} epochs, lr={lr})...")
    start = time.time()
    history = model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    end = time.time()

    elapsed = end - start
    history.history["train_time_sec"] = [elapsed]
    print(f"{phase} training complete in {elapsed / 60:.2f} min")

    return history


def plot_training_history(history, title_prefix, save_dir=PLOTS_DIR):
    """Plot training and validation accuracy/loss curves safely."""
    if not ENABLE_PLOTTING:
        return

    os.makedirs(save_dir, exist_ok=True)

    acc = history.history.get("accuracy") or history.history.get("acc") or []
    val_acc = history.history.get("val_accuracy") or history.history.get("val_acc") or []
    loss = history.history.get("loss") or []
    val_loss = history.history.get("val_loss") or []

    n_epochs = max(len(acc), len(val_acc), len(loss), len(val_loss))
    if n_epochs == 0:
        print(f"No metrics found to plot for {title_prefix}")
        return

    epochs = range(1, n_epochs + 1)

    if acc or val_acc:
        plt.figure(figsize=(8, 5))
        if acc:
            plt.plot(epochs[: len(acc)], acc, label="Training accuracy", color="tab:blue")
        if val_acc:
            plt.plot(
                epochs[: len(val_acc)],
                val_acc,
                label="Validation accuracy",
                color="tab:orange",
            )
        plt.title(f"{title_prefix} Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        if SAVE_PLOTS:
            acc_path = os.path.join(
                save_dir,
                f"{title_prefix.lower().replace(' ', '_')}_accuracy.png",
            )
            plt.savefig(acc_path)
            print(f"Saved accuracy plot -> {acc_path}")
        if SHOW_PLOTS:
            plt.show()
        plt.close()

    plt.figure(figsize=(8, 5))
    if loss:
        plt.plot(epochs[: len(loss)], loss, label="Training loss", color="tab:red")
    if val_loss:
        plt.plot(epochs[: len(val_loss)], val_loss, label="Validation loss", color="tab:green")
    plt.title(f"{title_prefix} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    if SAVE_PLOTS:
        loss_path = os.path.join(
            save_dir,
            f"{title_prefix.lower().replace(' ', '_')}_loss.png",
        )
        plt.savefig(loss_path)
        print(f"Saved loss plot -> {loss_path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def log_training_results(model_name, stage, history, total_time, lr, batch_size):
    """Append summary of a training stage to a CSV log file."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, LOG_FILE)

    best_acc = max(history.history.get("val_accuracy", [0]))
    best_loss = min(history.history.get("val_loss", [0]))

    headers = [
        "timestamp",
        "model",
        "stage",
        "epochs",
        "batch_size",
        "learning_rate",
        "best_val_accuracy",
        "best_val_loss",
        "total_time_sec",
    ]

    file_exists = os.path.isfile(log_path)
    with open(log_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                model_name,
                stage,
                len(history.epoch),
                batch_size,
                lr,
                round(best_acc, 4),
                round(best_loss, 4),
                round(total_time, 2),
            ]
        )

    print(f"Logged results for {model_name} ({stage}) -> {log_path}")


def unfreeze_top_layers(base_model, n):
    """Unfreeze the top N layers (excluding BatchNorms)."""
    base_model.trainable = True
    for layer in base_model.layers[:-n]:
        layer.trainable = False
    for layer in base_model.layers[-n:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    gpus = tf.config.list_physical_devices("GPU")
    print(f"Visible GPUs: {gpus}")

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    X_train_full, Y_train_full = preprocess_data(x_train, y_train)
    X_test, Y_test = preprocess_data(x_test, y_test)

    X_train, Y_train, X_val, Y_val = split_train_validation(X_train_full, Y_train_full)

    model, base_model = build_model(X_train.shape[1:])

    history1 = compile_and_train(
        model,
        LR_STAGE1,
        EPOCHS_STAGE1,
        X_train,
        Y_train,
        X_val,
        Y_val,
        "Stage 1 (frozen)",
    )
    plot_training_history(history1, "Stage 1 (Frozen)")
    log_training_results(
        model.name,
        "Stage 1 (Frozen)",
        history1,
        total_time=history1.history.get("train_time_sec", [0])[0],
        lr=LR_STAGE1,
        batch_size=BATCH_SIZE,
    )
    model.save(os.path.join(SAVE_DIR, f"{model.name}_stage1.keras"))

    unfreeze_top_layers(base_model, UNFREEZE_LAYERS)
    history2 = compile_and_train(
        model,
        LR_STAGE2,
        EPOCHS_STAGE2,
        X_train,
        Y_train,
        X_val,
        Y_val,
        "Stage 2 (fine-tuning)",
    )
    plot_training_history(history2, "Stage 2 (Fine-Tuning)")
    log_training_results(
        model.name,
        "Stage 2 (Fine-Tuning)",
        history2,
        total_time=history2.history.get("train_time_sec", [0])[0],
        lr=LR_STAGE2,
        batch_size=BATCH_SIZE,
    )

    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Final test accuracy: {test_acc:.4f} | test loss: {test_loss:.4f}")

    model.save(os.path.join(SAVE_DIR, f"{model.name}_finetuned.keras"))
    print("Fine-tuned model saved successfully!")
