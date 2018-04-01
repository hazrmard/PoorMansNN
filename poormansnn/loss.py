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
        pass


    def dEdy(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass



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
    TODO: Complete cross entropy loss
    The squared error loss. Mathematically:
    ```
    Error = sum (t * log(y))
    dError/dy = t / y
    ```
    """

    def __call__(self, t: np.ndarray, y: np.ndarray) -> float:
        return np.sum(-y * np.log(y))


    def dEdy(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        return t / y
