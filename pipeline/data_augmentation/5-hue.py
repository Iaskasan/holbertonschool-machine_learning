#!/usr/bin/env python3
"""Image hue adjustment utility."""

import tensorflow as tf


def change_hue(image, delta):
    """Adjust the hue of a 3D image tensor by delta."""
    return tf.image.adjust_hue(image, delta)
