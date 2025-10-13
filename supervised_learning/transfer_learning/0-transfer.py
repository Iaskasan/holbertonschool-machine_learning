#!/usr/bin/env python3
"""
Train a CNN to classify the CIFAR-10 dataset using a Keras Application.
"""

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os


def preprocess_data(X, Y):
    """Prepare CIFAR-10 data for training."""
    X = X.astype(np.float32, copy=False)
    np.multiply(X, 1.0 / 255.0, out=X, dtype=np.float32)
    if Y.ndim > 1:
        Y = Y.reshape(-1)
    Y = to_categorical(Y, num_classes=10)
    return X, Y


def build_model(input_shape):
    """
    Builds a CNN model using a pretrained Keras Application as base.

    Args:
        input_shape: tuple, shape of the input images

    Returns:
        model: compiled tf.keras Model
    """
    inputs = layers.Input(shape=input_shape)
    resized = layers.Lambda(lambda img: tf.image.resize(img, (224, 224)))(inputs)

    # Load ResNet50V2 model
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False, # remove old classification head
        weights='imagenet', # use pretrained weights
        input_tensor=resized # feed resized input
    )
    # freeze most of its layers
    base_model.trainable = False

    # add custom classification head (GlobalAveragePooling2D, Dense, etc.)
    x = layers.GlobalAveragePooling2D()(base_model.output) # GlobalAveragePooling instead of flatten since we want to classify
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile the model (Adam opt)
    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
    return model

def train_model(model, X_train, Y_train, X_val, Y_val):
    """
    Trains the model and returns the trained model.
    Also saves the plotting for accuracy and loss.

    Args:
        model: compiled tf.keras Model
        X_train, Y_train: training data
        X_val, Y_val: validation data

    Returns:
        trained model
    """
    # callbacks (Early stopping, checkpoint)
    callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
]

    # trains the model
    # store history of the model for later plotting
    history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=20,
    batch_size=64,
    callbacks=callbacks
)
    save_training_plots(history, model)
    return model

def save_training_plots(history, model, out_dir="images"):
    import os
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    name = model.name

    # --- Accuracy ---
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f"{name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{out_dir}/{name}_accuracy.png")
    plt.close()

    # --- Loss ---
    plt.figure()
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f"{name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{out_dir}/{name}_loss.png")
    plt.close()

    print(f"✅ Saved training graphs for {name} in '{out_dir}/'")

if __name__ == "__main__":
    tf.config.list_physical_devices('GPU')

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Preprocess the data
    X_train, Y_train = preprocess_data(x_train, y_train)
    X_test, Y_test = preprocess_data(x_test, y_test)

    # Build model
    model = build_model(X_train.shape[1:])

    # Train model
    model = train_model(model, X_train, Y_train, X_test, Y_test, plotting='True')

    # Save trained model
    model.save("cifar10.h5")

    # Adds visual of the history of the model (loss, accuracy, etc...)
