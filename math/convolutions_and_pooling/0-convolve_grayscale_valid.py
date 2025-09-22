#!/usr/bin/env python3
"""Performs a valid convolution on grayscale images"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    Args:
        images: numpy.ndarray (m, h, w), multiple grayscale images
        kernel: numpy.ndarray (kh, kw), the kernel for convolution

    Returns:
        numpy.ndarray: convolved images of shape (m, h-kh+1, w-kw+1)
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    output_h = h - kh + 1
    output_w = w - kw + 1
    output = np.zeros((m, output_h, output_w))
    for i in range(output_h):
        for j in range(output_w):
            region = images[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))
    return output
