#!/usr/bin/env python3
"""time-based decay of alpha"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay.

    Args:
        alpha (float): initial learning rate
        decay_rate (float): rate of decay
        global_step (int): number of gradient descent steps elapsed
        decay_step (int): number of steps before applying further decay

    Returns:
        float: updated learning rate
    """
    return alpha / (1 + decay_rate * (global_step // decay_step))
