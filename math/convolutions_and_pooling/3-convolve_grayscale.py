#!/usr/bin/env python3
"""performs a convolution on grayscale images"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images.

    images: numpy.ndarray of shape (m, h, w)
    kernel: numpy.ndarray of shape (kh, kw)
    padding: either a tuple of (ph, pw), 'same', or 'valid'
    stride: tuple of (sh, sw)

    Returns: numpy.ndarray of shape (m, h_new, w_new)
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph_top = ((h - 1) * sh + kh - h) // 2
        ph_bottom = ((h - 1) * sh + kh - h) - ph_top
        pw_left = ((w - 1) * sw + kw - w) // 2
        pw_right = ((w - 1) * sw + kw - w) - pw_left
    elif padding == 'valid':
        ph_top = ph_bottom = pw_left = pw_right = 0
    else:
        ph_top, pw_left = padding
        ph_bottom, pw_right = padding

    images_padded = np.pad(
        images,
        ((0, 0), (ph_top, ph_bottom), (pw_left, pw_right)),
        mode='constant',
        constant_values=0
    )

    h_new = (h + ph_top + ph_bottom - kh) // sh + 1
    w_new = (w + pw_left + pw_right - kw) // sw + 1

    output = np.zeros((m, h_new, w_new))

    for i in range(h_new):
        for j in range(w_new):
            patch = images_padded[:, i * sh:i * sh + kh, j * sw:j * sw + kw]
            output[:, i, j] = np.sum(patch * kernel, axis=(1, 2))

    return output
