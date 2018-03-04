"""
Implements gradient descent based optimization. Each optimizer has a __call__
method that takes a dim(weights) array of error derivatives w.r.t weights for each
N (inputs) x D (units) weights for a single layer. Returns a similar array
containing weight updates.
"""
from typing import Optional
import numpy as np


class Optimizer:

    def __init__(self, rate):
        self.rate = rate
    

    def pre(self, w: np.ndarray) -> Optional[np.ndarray]:
        pass


    def __call__(self, dEdw) -> np.ndarray:
        return -self.rate * dEdw



class Momentum(Optimizer):

    def __init__(self, rate, momentum=0.1):
        self.rate = rate
        self.momentum = momentum
        self.velocity = 0
    

    def pre(self, w: np.ndarray) -> Optional[np.ndarray]:
        pass
    

    def __call__(self, dEdw):
        self.velocity = self.momentum * self.velocity - self.rate * dEdw
        return self.velocity