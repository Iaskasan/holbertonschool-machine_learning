#!/usr/bin/env python3
"""updates the learning rate using inverse time decay"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ updates the learning rate using inverse time decay
    Args:
        alpha (float): original learning rate
        decay_rate (float): weight used to determine the rate at which
                           the learning rate decays
        global_step (int): number of passes of gradient descent that have
                           elapsed
        decay_step (int): number of passes of gradient descent that should
                          occur before updating the learning rate
    Returns:
        float: updated learning rate
    """
    lr = alpha / (1 + decay_rate * (global_step // decay_step))
    return lr
