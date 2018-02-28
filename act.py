"""
Contains activation functions for units and derivatives of activations w.r.t
their arguments. Each function accepts a numpy array N x D of N, D-dimensional
input instances. Returns an array of thte same shape.
"""
import numpy as np



# Linear/no activation
def linear(x: np.ndarray) -> np.ndarray:
    return x

def dlinear(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x)


# Hyperbolic tangent
def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def dtanh(x: np.ndarray) -> np.ndarray:
    return 1 - np.square(np.tanh(x))


# REctified Linear Unit
def relu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, 0)


def drelu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1., 0)


# Leaky REctified Linear Unit
def leaky_relu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, 1e-2*x)


def dlrelu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1., 1e-2)


# Sigmoid
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def dsigmoid(x: np.ndarray) -> np.ndarray:
    return x * (1 - x)
