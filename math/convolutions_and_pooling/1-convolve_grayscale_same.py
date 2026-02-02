#!/usr/bin/env python3
"""performs a same convolution on grayscale images"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

    images: numpy.ndarray of shape (m, h, w)
    kernel: numpy.ndarray of shape (kh, kw)

    Returns: numpy.ndarray of shape (m, h, w)
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    pad_top = kh // 2
    pad_bottom = kh - pad_top - 1
    pad_left = kw // 2
    pad_right = kw - pad_left - 1

    images_padded = np.pad(
        images,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=0
    )

    output = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            patch = images_padded[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(patch * kernel, axis=(1, 2))

    return output
