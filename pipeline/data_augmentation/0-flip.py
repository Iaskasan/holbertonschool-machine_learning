#!/usr/bin/env python3
"""Image flipping utility."""

import tensorflow as tf


def flip_image(image):
    """Flip a 3D image tensor horizontally."""
    return tf.image.flip_left_right(image)
