#!/usr/bin/env python3
"""Performs back propagation over a
convolutional layer of a neural network"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional
    layer of a neural network.

    Args:
        dZ: np.ndarray (m, h_new, w_new, c_new) - gradient of cost wrt Z
        A_prev: np.ndarray (m, h_prev, w_prev, c_prev) - prev layer activations
        W: np.ndarray (kh, kw, c_prev, c_new) - convolution kernels
        b: np.ndarray (1, 1, 1, c_new) - biases
        padding: 'same' or 'valid'
        stride: (sh, sw) - stride of convolution

    Returns:
        dA_prev, dW, db
        dA_prev: gradient wrt A_prev, shape (m, h_prev, w_prev, c_prev)
        dW: gradient wrt W, shape (kh, kw, c_prev, c_new)
        db: gradient wrt b, shape (1, 1, 1, c_new)
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride
    _, h_new, w_new, _ = dZ.shape
    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph, pw = 0, 0
    A_prev_pad = np.pad(A_prev,
                        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode="constant", constant_values=0)
    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw
                    a_slice = a_prev_pad[vert_start:vert_end,
                                         horiz_start:horiz_end, :]
                    da_prev_pad[vert_start:vert_end,
                                horiz_start:horiz_end,
                                :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]

        dA_prev_pad[i] = da_prev_pad
    if padding == "same":
        dA_prev = dA_prev_pad[:, ph:-ph if ph else None,
                              pw:-pw if pw else None, :]
    else:
        dA_prev = dA_prev_pad

    return dA_prev, dW, db
