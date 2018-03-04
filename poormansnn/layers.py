"""
Implements layers is modular units in a network. Each layer is able to:

* feedforward inputs through activations and return output,
* Return error derivatives with respect to:
  * the argument (z) of the activation function, dE/dz
  * the the weights (w) into the layer, dE/dw
  * the biases (b) into the layer, dE/db
  * the inputs (x) into the layer, dE/dx
"""

from typing import Iterable
import numpy as np
from .act import Activation


class Layer:
    """
    A fully connected layer. A layer is defined by:
    * **a**: The layer activation i.e. output for each unit in layer,
    * **z**: The input to the activation function for each unit in layer,
    * **w**: Weights connecting each unit to inputs/previous layer outputs,
    * **x**: Inputs/previous layer outputs,
    * **b**: Bias for each unit in layer.

    Mathematically:
    ```
    z = w*x + b
    a = act(z)
    ```

    Args:
    * shape (Iterable[int]): The dimensions of the layer.
    * act (Activation): The activation function and derivative.

    Attributes:
    * **W** (np.ndarray): A dim(x) x dim(a) array of weights connecting previous
    layer to current layer.
    * **b** (np.ndarray): A dim(a) array of bias values for each unit in layer.
    """

    def __init__(self, shape: Iterable[int], prevshape: Iterable[int], act: Activation):
        self.shape = np.array(shape)
        self.prevshape = np.array(prevshape)
        self.act = act

        self.z: np.ndarray = None
        self.a: np.ndarray = None
        self.x: np.ndarray = None
        self.w = np.random.rand(*self.prevshape, *self.shape) \
                * 2 / np.sqrt(sum(self.prevshape)) \
                - np.sqrt(sum(self.shape))
        self.b = np.zeros(self.shape)




    def feedforward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        # N x dim(a) = N x dim(x) * dim(x) x dim(a)
        self.z = np.tensordot(x, self.w, axes=len(x.shape)-1)
        self.a = self.act.act(self.z)
        return self.a



    def dEdz(self, dEda: np.ndarray) -> np.ndarray:
        # dE/dz = dE/da * da/dz
        # N x dim(a) = N x dim(a) .* N x dim(a)
        return self.act.dadz(self.a, self.z) * dEda



    def dEdw(self, dEdz: np.ndarray) -> np.ndarray:
        # dE/dw = dz/dw * dE/dz
        # dE/dw = x * dE/dz
        # dim(x) x dim(a) = N x dim(x) * N x dim(a)
        return np.tensordot(self.x, dEdz, axes=[[0], [0]]) / len(self.x)



    def dEdb(self, dEdz: np.ndarray) -> np.ndarray:
        # dE/dw = dE/dz * dz/db
        # dE/dw = dE/dz * 1
        # dim(a) = mean(N x dim(a))
        return np.mean(dEdz, 0)



    def dEdx(self, dEdz: np.ndarray) -> np.ndarray:
        # dE/dx = dE/dz * dz/dx
        # dE/dx = dE/dz * w
        # N x dim(x) = N x dim(a) * dim(x) x dim(a)
        return np.tensordot(dEdz, self.w,
                axes=[np.arange(1, len(dEdz.shape)),
                      np.arange(len(self.prevshape), len(self.w.shape))])