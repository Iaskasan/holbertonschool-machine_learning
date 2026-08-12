#!/usr/bin/env python3
"""Method for Image brightness change."""

import tensorflow as tf


def change_brightness(image, max_delta):
    """Randomly adjust the brightness of a 3D image tensor."""
    return tf.image.random_brightness(image, max_delta=max_delta)
