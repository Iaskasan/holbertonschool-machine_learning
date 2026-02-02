#!/usr/bin/env python3
"""performs a valid convolution on grayscale images"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    images: numpy.ndarray of shape (m, h, w)
    kernel: numpy.ndarray of shape (kh, kw)

    Returns:
        numpy.ndarray of shape (m, h - kh + 1, w - kw + 1)
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    out_h = h - kh + 1
    out_w = w - kw + 1

    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            patch = images[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(patch * kernel, axis=(1, 2))

    return output
