#!/usr/bin/env python3
"""Convolution on images with channels"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels.

    Args:
        images (numpy.ndarray): shape (m, h, w, c)
            - m: number of images
            - h: height of each image
            - w: width of each image
            - c: number of channels
        kernel (numpy.ndarray): shape (kh, kw, c)
            - kh: height of the kernel
            - kw: width of the kernel
            - c: must match the number of channels in images
        padding (str or tuple): 'same', 'valid', or (ph, pw)
            - 'same': output size matches input size
            - 'valid': no padding
            - (ph, pw): custom padding for height and width
        stride (tuple): (sh, sw)
            - sh: stride along height
            - sw: stride along width

    Returns:
        numpy.ndarray: convolved images of shape (m, out_h, out_w)
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = max(((h - 1) * sh + kh - h) // 2 + 1, 0)
        pw = max(((w - 1) * sw + kw - w) // 2 + 1, 0)
    elif padding == 'valid':
        ph = pw = 0
    else:
        ph, pw = padding
    padded = np.pad(images,
                    ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode='constant')
    out_h = (padded.shape[1] - kh) // sh + 1
    out_w = (padded.shape[2] - kw) // sw + 1
    output = np.zeros((m, out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            region = padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2, 3))
    return output
