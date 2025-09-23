#!/usr/bin/env python3
"""Performs a convolution on grayscale images"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images.

    Args:
        images: numpy.ndarray (m, h, w), grayscale images
        kernel: numpy.ndarray (kh, kw), kernel for convolution
        padding: tuple (ph, pw), 'same', or 'valid'
        stride: tuple (sh, sw), stride along height and width

    Returns:
        numpy.ndarray: convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride
    if type(padding) is tuple:
        ph, pw = padding
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    padded = np.pad(images,
                    pad_width=((0, 0), (ph, ph), (pw, pw)),
                    mode='constant')
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    output = np.zeros((m, out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            i_start, j_start = i * sh, j * sw
            region = padded[:, i_start:i_start+kh, j_start:j_start+kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))
    return output
