#!/usr/bin/env python3
"""
Fine-tune EfficientNetV2S on CIFAR-10 using the previously trained model.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import csv
from datetime import datetime


# Tweaking section, here you can modify saving directories, verbose, fine tuning parameters...
MODEL_NAME = "efficientnetv2s_20ep" # this will be the prefix of the saved files
BASE_MODEL = tf.keras.applications.EfficientNetV2S
TARGET_SIZE = (224, 224)

EPOCHS_STAGE1 = 20        # initial frozen training
EPOCHS_STAGE2 = 15        # fine-tuning phase
BATCH_SIZE = 128
EARLY_STOPPING = 3       # Early-stopping patience value

LR_STAGE1 = 1e-4          # learning rate for head training
LR_STAGE2 = 1e-6          # lower learning rate for fine-tuning

UNFREEZE_LAYERS = 20      # number of top layers to unfreeze
DROPOUT1 = 0.5
DROPOUT2 = 0.3
DENSE1 = 512
DENSE2 = 256
PREPROCESSING = False     # Some models have in-built preprocessing,
                          # set this to False if you want to use the custom preprocessing

USE_MIXED_PRECISION = True
SAVE_DIR = f"trained_models/{MODEL_NAME}"
PLOTS_DIR = f"training_plots/{MODEL_NAME}"
LOGS_DIR = "logs"
LOG_FILE = "training_results.csv"

ENABLE_PLOTTING = True       # toggle plotting ON/OFF
SAVE_PLOTS = True            # save plots to file
SHOW_PLOTS = False           # display with plt.show() interactively


if USE_MIXED_PRECISION:
    mixed_precision.set_global_policy('mixed_float16')


def preprocess_data(X, Y):
    X = X.astype("float32") / 255.0
    Y = to_categorical(Y, 10)
    return X, Y


def build_model(input_shape):
    """Builds a CNN model using a pretrained Keras Application as base."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Resizing(*TARGET_SIZE)(inputs)

    base_model = BASE_MODEL(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
        include_preprocessing=PREPROCESSING
    )
    base_model.trainable = False  # frozen for stage 1

    # Classification head
    y = layers.GlobalAveragePooling2D()(base_model.output)
    y = layers.BatchNormalization()(y)
    y = layers.Dense(DENSE1, activation="relu")(y)
    y = layers.Dropout(DROPOUT1)(y)
    y = layers.Dense(DENSE2, activation="relu")(y)
    y = layers.Dropout(DROPOUT2)(y)
    outputs = layers.Dense(10, activation="softmax", dtype="float32")(y)

    model = models.Model(inputs, outputs, name=f"cifar10_{MODEL_NAME}")
    return model, base_model


def compile_and_train(model, lr, epochs, X_train, Y_train, X_val, Y_val, phase):
    """Compile and train with given hyperparameters."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=EARLY_STOPPING, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=1, min_lr=1e-6)
    ]

    print(f"\n🚀 Starting {phase} training ({epochs} epochs, lr={lr})...")
    start = time.time()
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    end = time.time()
    print(f"✅ {phase} training complete in {(end - start)/60:.2f} min")

    return history

def plot_training_history(history, title_prefix, save_dir=PLOTS_DIR):
    """Plots training and validation accuracy/loss curves safely."""
    if not ENABLE_PLOTTING:
        return

    os.makedirs(save_dir, exist_ok=True)

    # Try to get all possible metric names safely
    acc = history.history.get('accuracy') or history.history.get('acc') or []
    val_acc = history.history.get('val_accuracy') or history.history.get('val_acc') or []
    loss = history.history.get('loss') or []
    val_loss = history.history.get('val_loss') or []

    # Determine number of epochs based on whichever list is non-empty
    n_epochs = max(len(acc), len(val_acc), len(loss), len(val_loss))
    if n_epochs == 0:
        print(f"⚠️ No metrics found to plot for {title_prefix}")
        return

    epochs = range(1, n_epochs + 1)

    # --- Accuracy plot ---
    if acc or val_acc:  # only plot if accuracy data exists
        plt.figure(figsize=(8, 5))
        if acc:
            plt.plot(epochs[:len(acc)], acc, label='Training accuracy', color='tab:blue')
        if val_acc:
            plt.plot(epochs[:len(val_acc)], val_acc, label='Validation accuracy', color='tab:orange')
        plt.title(f'{title_prefix} Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        if SAVE_PLOTS:
            acc_path = os.path.join(save_dir, f"{title_prefix.lower().replace(' ', '_')}_accuracy.png")
            plt.savefig(acc_path)
            print(f"✅ Saved accuracy plot → {acc_path}")
        if SHOW_PLOTS:
            plt.show()
        plt.close()

    # --- Loss plot ---
    plt.figure(figsize=(8, 5))
    if loss:
        plt.plot(epochs[:len(loss)], loss, label='Training loss', color='tab:red')
    if val_loss:
        plt.plot(epochs[:len(val_loss)], val_loss, label='Validation loss', color='tab:green')
    plt.title(f'{title_prefix} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    if SAVE_PLOTS:
        loss_path = os.path.join(save_dir, f"{title_prefix.lower().replace(' ', '_')}_loss.png")
        plt.savefig(loss_path)
        print(f"✅ Saved loss plot → {loss_path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def log_training_results(model_name, stage, history, total_time, lr, batch_size):
    """Append summary of a training stage to a CSV log file."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, LOG_FILE)

    # extract best metrics
    best_acc = max(history.history.get('val_accuracy', [0]))
    best_loss = min(history.history.get('val_loss', [0]))

    # CSV headers
    headers = [
        "timestamp", "model", "stage",
        "epochs", "batch_size", "learning_rate",
        "best_val_accuracy", "best_val_loss",
        "total_time_sec"
    ]

    file_exists = os.path.isfile(log_path)
    with open(log_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_name, stage,
            len(history.epoch), batch_size, lr,
            round(best_acc, 4), round(best_loss, 4),
            round(total_time, 2)
        ])

    print(f"📝 Logged results for {model_name} ({stage}) → {log_path}")

def unfreeze_top_layers(base_model, n):
    """Unfreeze the top N layers (excluding BatchNorms)."""
    base_model.trainable = True
    for layer in base_model.layers[:-n]:
        layer.trainable = False
    for layer in base_model.layers[-n:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


# Main training logic


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    X_train, Y_train = preprocess_data(x_train, y_train)
    X_test, Y_test = preprocess_data(x_test, y_test)

    # 1️⃣ Stage 1 — Train classification head
    model, base_model = build_model(X_train.shape[1:])
    history1 = compile_and_train(
        model, LR_STAGE1, EPOCHS_STAGE1, X_train, Y_train, X_test, Y_test, "Stage 1 (frozen)"
    )
    plot_training_history(history1, "Stage 1 (Frozen)") # Plotting method
    log_training_results(model.name, "Stage 1 (Frozen)", history1,
                     total_time=sum(history1.history.get("time", [])) if "time" in history1.history else 0,
                     lr=LR_STAGE1, batch_size=BATCH_SIZE)
    model.save(os.path.join(SAVE_DIR, f"{model.name}_stage1.h5"))

    # 2️⃣ Stage 2 — Fine-tune last N layers
    unfreeze_top_layers(base_model, UNFREEZE_LAYERS)
    history2 = compile_and_train(
        model, LR_STAGE2, EPOCHS_STAGE2, X_train, Y_train, X_test, Y_test, "Stage 2 (fine-tuning)"
    )
    plot_training_history(history2, "Stage 2 (Fine-Tuning)")
    log_training_results(model.name, "Stage 2 (Fine-Tuning)", history2,
                     total_time=0,  # optional if you don't time stage2 separately
                     lr=LR_STAGE2, batch_size=BATCH_SIZE)
    model.save(os.path.join(SAVE_DIR, f"{model.name}_finetuned.h5"))
    print("🎉 Fine-tuned model saved successfully!")
