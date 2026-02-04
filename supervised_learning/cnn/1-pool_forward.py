#!/usr/bin/env python3
"""Performs forward propagation over a
pooling layer of a neural network"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling
    layer of a neural network.

    Args:
        A_prev: np.ndarray of shape (m, h_prev, w_prev, c_prev)
                output of the previous layer
        kernel_shape: tuple (kh, kw), size of the pooling kernel
        stride: tuple (sh, sw), strides for pooling
        mode: 'max' or 'avg', type of pooling

    Returns:
        np.ndarray containing the output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    h_out = int((h_prev - kh) / sh) + 1
    w_out = int((w_prev - kw) / sw) + 1
    A_pool = np.zeros((m, h_out, w_out, c_prev))

    for i in range(h_out):
        for j in range(w_out):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw
            A_slice = A_prev[:, vert_start:vert_end, horiz_start:horiz_end, :]
            if mode == 'max':
                A_pool[:, i, j, :] = np.max(A_slice, axis=(1, 2))
            elif mode == 'avg':
                A_pool[:, i, j, :] = np.mean(A_slice, axis=(1, 2))

    return A_pool
