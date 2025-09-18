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
        tf.Tensor: tensor of total costs, one per layer
    """
    l2_losses = model.losses
    total_costs = [cost + l2 for l2 in l2_losses]
    return tf.convert_to_tensor(total_costs)
