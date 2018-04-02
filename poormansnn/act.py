# TODO: return multidimensional gradients for each activation w.r.t each logit.
# The gradient is a N x dim(a) x dim(z) matrix (dim(a)==dim(z))
"""
Defines the Activation class which contains activation functions for units and
derivatives of activations w.r.t their arguments. Each class has a signature of
two methods:

* __call__(z): Takes an array of arguments to activation function and returns a
similarly shaped array.
* dadz(a, z): Takes 2 arrays of the activation(a) and the activation arguments(z)
and returns a similarly shaped array of the derivative da/dz.
"""
import numpy as np



class Activation:
    """
    The activation function and its derivative. An activation is defined by:
    * **a**: The output of the activation function,
    * **z**: The input to the activation function.

    Mathematically:
    ```
    a = act(z)
    ```
    """

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return z

    def dadz(self, a: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.ones_like(z)



class Linear(Activation):

    pass


class Tanh(Activation):

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)


    def dadz(self, a: np.ndarray, z: np.ndarray) -> np.ndarray:
        return 1 - np.square(a)



class Relu(Activation):
        
    def __call__(self, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, z, 0)


    def dadz(self, a: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, 1., 0)




class Leaky_relu(Activation):

    def __init__(self, alpha: float = 1e-3):
        self.alpha = alpha
        
    def __call__(self, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, z, self.alpha*z)


    def dadz(self, a: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, 1., self.alpha)




class Sigmoid(Activation):
        
    def __call__(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))


    def dadz(self, a: np.ndarray, z: np.ndarray) -> np.ndarray:
        return a * (1 - a)



class Softmax(Activation):
    """
    Only works with cross-entropy loss for single-label multi-class problems.
    i.e. only one output unit can have label=1 at a time.
    """

    def __call__(self, z: np.ndarray) -> np.ndarray:
        zexp = np.exp(z - z.min())
        zsum = zexp.sum(axis=tuple(range(1, z.ndim)))
        return zexp / zsum.reshape(len(zexp), *([1] * (z.ndim-1)))     
    

    def dadz(self, a: np.ndarray, z: np.ndarray) -> np.ndarray:
        # TODO: finish
        return a * (1 - a)
    

    def dadz_softmax(self, a: np.ndarray, z: np.ndarray) -> np.ndarray:
        # the whole calculation for dE/dz is done by loss.CrossEntropyLoss.dEdz_softmax
        return np.ndarray([1])