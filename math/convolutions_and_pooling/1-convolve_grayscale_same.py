#!/usr/bin/env python3
"""Performs a same convolution on grayscale images"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

    Args:
        images: numpy.ndarray (m, h, w), multiple grayscale images
        kernel: numpy.ndarray (kh, kw), the kernel for convolution

    Returns:
        numpy.ndarray: convolved images of shape (m, h, w)
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    pad_top = kh // 2
    pad_bottom = kh - 1 - pad_top
    pad_left = kw // 2
    pad_right = kw - 1 - pad_left
    padded = np.pad(images,
                    pad_width=((0, 0),
                               (pad_top, pad_bottom),
                               (pad_left, pad_right)),
                    mode='constant')
    output = np.zeros((m, h, w))
    for i in range(h):
        for j in range(w):
            region = padded[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))
    return output
