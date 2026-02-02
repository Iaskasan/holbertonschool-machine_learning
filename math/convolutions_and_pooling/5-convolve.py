#!/usr/bin/env python3
"""Multi-kernel convolution on images"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels.

    Args:
        images (numpy.ndarray): shape (m, h, w, c)
            - m: number of images
            - h: height of each image
            - w: width of each image
            - c: number of channels in the image
        kernels (numpy.ndarray): shape (kh, kw, c, nc)
            - kh: height of the kernel
            - kw: width of the kernel
            - c: must match image channels
            - nc: number of kernels (feature maps)
        padding (str or tuple): 'same', 'valid', or (ph, pw)
            - 'same': output size matches input size
            - 'valid': no padding
            - (ph, pw): custom padding for height and width
        stride (tuple): (sh, sw)
            - sh: stride along height
            - sw: stride along width

    Returns:
        numpy.ndarray: convolved output of shape (m, out_h, out_w, nc)
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
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
    output = np.zeros((m, out_h, out_w, nc))
    for i in range(out_h):
        for j in range(out_w):
            region = padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            for k in range(nc):
                kernel = kernels[:, :, :, k]
                output[:, i, j, k] = np.sum(region * kernel, axis=(1, 2, 3))
    return output
