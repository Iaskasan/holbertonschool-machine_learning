#!/usr/bin/env python3
"""performs a convolution on grayscale images
with optional padding and stride"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a 2D convolution on grayscale images with optional
    padding and stride.

    Args:
        images (numpy.ndarray): Input images of shape (m, h, w).
            - m: number of images
            - h: height of each image
            - w: width of each image
        kernel (numpy.ndarray): Convolution kernel of shape (kh, kw).
            - kh: height of the kernel
            - kw: width of the kernel
        padding (str or tuple): Type of padding to apply.
            - 'same': output has the same spatial dimensions as input
            - 'valid': no padding is applied
            - (ph, pw): tuple of ints giving padding for height and width
        stride (tuple): The stride of the convolution, (sh, sw).
            - sh: stride along height
            - sw: stride along width

    Returns:
        numpy.ndarray: Convolved output of shape (m, output_h, output_w)
            - output_h: height of the output
            - output_w: width of the output

    Notes:
        - The convolution is applied independently to each image.
        - Padding is done with zeros.
        - This function does not use built-in convolution methods;
          it explicitly performs the operation using numpy.

    Example:
        >>> images = np.random.randint(0, 256, (2, 5, 5))
        >>> kernel = np.array([[1, 0], [0, -1]])
        >>> convolve_grayscale(images, kernel, padding='valid',
        stride=(1, 1)).shape
        (2, 4, 4)
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride
    if padding == 'same':
        ph = ((h - 1) * sh + kh - h + 1) // 2
        pw = ((w - 1) * sw + kw - w + 1) // 2
    elif padding == 'valid':
        ph = pw = 0
    else:
        ph, pw = padding
    if ph > 0 or pw > 0:
        padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                               mode='constant', constant_values=0)
    else:
        padded_images = images
    padded_h, padded_w = padded_images.shape[1], padded_images.shape[2]
    output_h = (padded_h - kh) // sh + 1
    output_w = (padded_w - kw) // sw + 1
    output = np.zeros((m, output_h, output_w))
    for i in range(output_h):
        for j in range(output_w):
            start_i = i * sh
            start_j = j * sw
            output[:, i, j] = np.sum(
                padded_images[:, start_i:start_i+kh,
                              start_j:start_j+kw] * kernel,
                axis=(1, 2)
            )
    return output
