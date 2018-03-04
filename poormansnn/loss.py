"""
Error functions and their derivatives w.r.t predicted outputs. Each class has
a signature of two methods:

* __call__(y, ypred): Takes a N x dim(output) array of labels and
predictions and returns a float representing the error measure/total loss.

* dEdy(y, ypred): Takes and returns a N x dim(output) array of error
derivative w.r.t predictions i.e. dE/dypred.

Where `y` is the true label and `ypred` is the prediction.
"""

import numpy as np



class Loss:
    """
    The ptototypical loss class. Implements two methods:
    * __call__(labels: np.ndarray, predictions: np.ndarray) -> float
    * dEdy(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray
    """

    def __call__(self, y: np.ndarray, ypred: np.ndarray) -> float:
        pass


    def dEdy(self, y: np.ndarray, ypred: np.ndarray) -> np.ndarray:
        pass



class SquaredLoss(Loss):
    """
    The squared error loss. Mathematically:
    ```
    error = 0.5 * (labels - predictions)^2
    dE/dy = (predictions - labels)
    ```
    """

    def __call__(self, y: np.ndarray, ypred: np.ndarray) -> float:
        return np.sum(np.mean(0.5 * np.square(ypred - y), 0))


    def dEdy(self, y: np.ndarray, ypred: np.ndarray) -> np.ndarray:
        return ypred - y
