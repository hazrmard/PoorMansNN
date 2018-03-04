"""
Implements gradient descent based optimization. Each optimizer implements:
* pre(w: np.ndarray): Takes a dim(w) array of weights for a single layer. Makes
changes to weights before computation of derivatives.
* __call__(dEdw: np.ndarray): Takes a dim(w) array of error derivatives w.r.t
weights for a single layer. Returns a similar array containing weight updates.
NOTE: Not implemented yet.
"""
from typing import Optional
import numpy as np


class Optimizer:
    """
    The standard stochastic gradient descent optimizer. Mathematically:
    ```
    delta(w) = -learning_rate * dE/dw
    ```
    """

    def __init__(self, rate):
        self.rate = rate
    

    def pre(self, w: np.ndarray) -> Optional[np.ndarray]:
        pass


    def __call__(self, dEdw) -> np.ndarray:
        return -self.rate * dEdw



class Momentum(Optimizer):
    """
    The momentum optimizer. Mathematically:
    ```
    velocity = momentum * velocity - learning_rate * dE/dw
    delta(w) = velocity
    ```
    """

    def __init__(self, rate, momentum=0.1):
        self.rate = rate
        self.momentum = momentum
        self.velocity = 0
    

    def pre(self, w: np.ndarray) -> Optional[np.ndarray]:
        pass
    

    def __call__(self, dEdw):
        self.velocity = self.momentum * self.velocity - self.rate * dEdw
        return self.velocity