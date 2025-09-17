#!/usr/bin/env python3
"""L2 Regularization Cost"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization.

    Args:
        cost (tf.Tensor): base cost of the network (without L2 reg).
        model (tf.keras.Model): model with L2-regularized layers.

    Returns:
        tf.Tensor: total cost including L2 regularization terms.
    """
    l2_loss = tf.add_n(model.losses)
    total_cost = cost + l2_loss
    return total_cost
