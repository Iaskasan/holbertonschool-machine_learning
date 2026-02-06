#!/usr/bin/env python3
"""Modified version of the LeNet-5 using keras"""
from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified version of LeNet-5 architecture using Keras.

    Args:
        X: K.Input of shape (m, 28, 28, 1) containing input images

    Returns:
        A compiled K.Model
    """
    initializer = K.initializers.HeNormal(seed=0)

    # Layer 1: Conv -> ReLU
    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding="same",
        activation="relu",
        kernel_initializer=initializer
    )(X)

    # Layer 2: Max Pool
    pool1 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv1)

    # Layer 3: Conv -> ReLU
    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding="valid",
        activation="relu",
        kernel_initializer=initializer
    )(pool1)

    # Layer 4: Max Pool
    pool2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv2)

    # Flatten before Dense layers
    flat = K.layers.Flatten()(pool2)

    # Layer 5: Fully Connected (120) -> ReLU
    fc1 = K.layers.Dense(
        units=120,
        activation="relu",
        kernel_initializer=initializer
    )(flat)

    # Layer 6: Fully Connected (84) -> ReLU
    fc2 = K.layers.Dense(
        units=84,
        activation="relu",
        kernel_initializer=initializer
    )(fc1)

    # Layer 7: Fully Connected (10) -> Softmax
    output = K.layers.Dense(
        units=10,
        activation="softmax",
        kernel_initializer=initializer
    )(fc2)

    # Build and compile model
    model = K.Model(inputs=X, outputs=output)
    model.compile(
        optimizer=K.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
