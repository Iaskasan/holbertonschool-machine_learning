#!/usr/bin/env python3
"""performs a convolution on grayscale images with custom padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.

    images: numpy.ndarray of shape (m, h, w)
    kernel: numpy.ndarray of shape (kh, kw)
    padding: tuple of (ph, pw)

    Returns: numpy.ndarray of shape (m, h + 2*ph - kh + 1, w + 2*pw - kw + 1)
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    images_padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant',
        constant_values=0
    )

    out_h = h + 2 * ph - kh + 1
    out_w = w + 2 * pw - kw + 1

    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            patch = images_padded[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(patch * kernel, axis=(1, 2))

    return output
