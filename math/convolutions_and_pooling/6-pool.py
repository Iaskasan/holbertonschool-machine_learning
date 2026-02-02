#!/usr/bin/env python3
"""performs a pooling on images"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.

    Args:
        images (numpy.ndarray): shape (m, h, w, c)
            - m: number of images
            - h: height of each image
            - w: width of each image
            - c: number of channels in the image
        kernel_shape (tuple): (kh, kw) pooling window size
        stride (tuple): (sh, sw) stride size
        mode (str): pooling mode
            - 'max': max pooling
            - 'avg': average pooling

    Returns:
        numpy.ndarray: pooled images of shape (m, out_h, out_w, c)
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1
    pooled = np.zeros((m, out_h, out_w, c))
    for i in range(out_h):
        for j in range(out_w):
            region = images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            if mode == 'max':
                pooled[:, i, j, :] = np.max(region, axis=(1, 2))
            elif mode == 'avg':
                pooled[:, i, j, :] = np.mean(region, axis=(1, 2))
            else:
                raise ValueError("mode must be 'max' or 'avg'")
    return pooled
