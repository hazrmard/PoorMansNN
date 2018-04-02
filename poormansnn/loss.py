"""
Error functions and their derivatives w.r.t predicted outputs. Each class has
a signature of two methods:

* __call__(t, y): Takes a N x dim(output) array of labels and
predictions and returns a float representing the error measure/total loss.

* dEdy(t, y): Takes and returns a N x dim(output) array of error
derivative w.r.t predictions i.e. dE/dy.

Where `t` is the true/target label and `y` is the prediction.
"""

import numpy as np



class Loss:
    """
    The ptototypical loss class. Implements two methods:
    * __call__(labels: np.ndarray, predictions: np.ndarray) -> float
    * dEdy(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray
    """

    def __call__(self, t: np.ndarray, y: np.ndarray) -> float:
        return np.sum(np.abs(t - y))


    def dEdy(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        np.where(y > t, 1, -1)



class SquaredLoss(Loss):
    """
    The squared error loss. Mathematically:
    ```
    Error = 0.5 * (t - y)^2
    dError/dy = (y - t)
    ```
    """

    def __call__(self, t: np.ndarray, y: np.ndarray) -> float:
        return np.sum(np.mean(0.5 * np.square(t - y), 0))


    def dEdy(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        return y - t



class CrossEntropyLoss(Loss):
    """
    The squared error loss. Mathematically:
    ```
    Error = sum (t * log(y))
    dError/dy = t / y
    ```
    """

    def __init__(self, softmax: bool=True):
        self.is_softmax = True


    def __call__(self, t: np.ndarray, y: np.ndarray) -> float:
        mask = t > 0
        return np.sum(-t[mask] * np.log(y[mask]))


    def dEdy(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        return t / y
    

    def dEda_softmax(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        # since dE/dz = dE/dy * dy/dz = (t / y) * y * (1 - y) = t * (1 - y)
        # foregoes division by 'y' in case of divide-by-zero
        # instead returns one half of the expression, the other half is returned
        # by act.Softmax.dadz_softmax
        print(t)
        label = t[t!=0][0]
        return np.where(t==0, -label * y, label * (1 - y))
