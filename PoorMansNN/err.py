"""
Error functions and their derivatives w.r.t predicted outputs. Each function
accepts two arguments: true labels, and predicted labels. Both arguments are
N x D arrays of N, D-dimensional labels and predictions.

The error function returns a float of the mean loss over instances.

The error derivative returns a N x D array containing derivative of the error
for each of D output units, and for each of N instances.
"""

import numpy as np


# Squared error
def squared(y, ypred):
    return np.sum(np.mean(0.5 * np.square(ypred - y), 0))


def dsquared(y, ypred):
    return ypred - y
