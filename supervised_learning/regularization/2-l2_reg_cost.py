#!/usr/bin/env python3
"""calculates the cost of a neural network with L2 regularization"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates total cost with L2 regularization for a Keras model

    Args:
        cost: tensor, cost without L2 regularization
        model: keras model with L2 regularization

    Returns:
        tensor, total cost including L2 regularization
    """
    return cost + tf.stack(model.losses)
