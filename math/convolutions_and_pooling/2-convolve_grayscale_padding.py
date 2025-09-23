#!/usr/bin/env python3
"""Performs a convolution on grayscale images with custom padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.

    Args:
        images: numpy.ndarray (m, h, w), grayscale images
        kernel: numpy.ndarray (kh, kw), the kernel for convolution
        padding: tuple (ph, pw), custom padding size

    Returns:
        numpy.ndarray: convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    padded = np.pad(images,
                    pad_width=((0, 0), (ph, ph), (pw, pw)),
                    mode='constant')
    output_h = h + 2*ph - kh + 1
    output_w = w + 2*pw - kw + 1
    output = np.zeros((m, output_h, output_w))
    for i in range(output_h):
        for j in range(output_w):
            region = padded[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))
    return output
