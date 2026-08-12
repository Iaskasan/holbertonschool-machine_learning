#!/usr/bin/env python3
"""Image cropping utility."""

import tensorflow as tf


def crop_image(image, size):
    """Randomly crop a 3D image tensor to the specified size."""
    return tf.image.random_crop(image, size=size)
