#!/usr/bin/env python3
"""Image contrast adjustment utility."""

import tensorflow as tf


def change_contrast(image, lower, upper):
    """Randomly adjust the contrast of a 3D image tensor."""
    return tf.image.random_contrast(image, lower=lower, upper=upper)
