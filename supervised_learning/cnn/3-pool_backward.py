#!/usr/bin/env python3
"""Performs back propagation over a pooling layer of a neural network"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network.

    Args:
        dA: np.ndarray (m, h_new, w_new, c) - gradient wrt pooling output
        A_prev: np.ndarray (m, h_prev, w_prev, c) - input to pooling layer
        kernel_shape: tuple (kh, kw) - pooling kernel size
        stride: tuple (sh, sw) - stride of pooling
        mode: 'max' or 'avg'

    Returns:
        dA_prev: np.ndarray (m, h_prev, w_prev, c) - gradient wrt A_prev
    """
    m, h_prev, w_prev, c = A_prev.shape
    m, h_new, w_new, c = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride
    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for channel in range(c):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw
                    if mode == 'max':
                        a_slice = A_prev[i, vert_start:vert_end,
                                         horiz_start:horiz_end, channel]
                        mask = (a_slice == np.max(a_slice))
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end,
                                channel] += mask * dA[i, h, w, channel]
                    elif mode == 'avg':
                        average = dA[i, h, w, channel] / (kh * kw)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end,
                                channel] += np.ones((kh, kw)) * average

    return dA_prev
