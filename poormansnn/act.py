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

    @staticmethod
    def diag_idx(arr: np.ndarray):
        # TODO: correct implementation
        """
        Calculates indices of diagonal elements in each batch in array where the
        first dimension is along batches and the rest are square.
        For e.g N x d x d x d x ...
        """
        ndim = arr.ndim - 1
        batchsize = arr.shape[0]
        dim = arr.shape[1]  # assuming all except first dimensions are same
        idx = np.diag_indices(dim, ndim)
        idx = np.tile(idx, batchsize) # ndim x dim*batchsize
        batchidx = np.repeat(np.arange(batchsize), dim).reshape((1, dim*batchsize))
        indices = np.concatenate((batchidx, idx), axis=0)
        return [x for x in indices]
        


    def __call__(self, z: np.ndarray) -> np.ndarray:
        return z

    def dadz(self, a: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.ones_like(z)
        # dadz = np.zeros((*z.shape, *a.shape[1:]))
        # dadz[self.diag_idx(dadz)] = 1
        # return dadz



class Linear(Activation):

    pass


class Tanh(Activation):

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)


    def dadz(self, a: np.ndarray, z: np.ndarray) -> np.ndarray:
        return 1 - np.square(a)
        # dadz = super().dadz(a, z)
        # dadz[self.diag_idx(dadz)] = (1 - np.square(a)).reshape(-1)
        # return dadz



class Relu(Activation):
        
    def __call__(self, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, z, 0)


    def dadz(self, a: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, 1., 0)
        # dadz = super().dadz(a, z)
        # dadz[self.diag_idx(dadz)] = np.where(z > 0, 1., 0).reshape(-1)
        # return dadz




class Leaky_relu(Activation):

    def __init__(self, alpha: float = 1e-3):
        self.alpha = alpha
        
    def __call__(self, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, z, self.alpha*z)


    def dadz(self, a: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, 1., self.alpha)
        # dadz = super().dadz(a, z)
        # dadz[self.diag_idx(dadz)] = np.where(z > 0, 1., self.alpha).reshape(-1)
        # return dadz




class Sigmoid(Activation):
        
    def __call__(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))


    def dadz(self, a: np.ndarray, z: np.ndarray) -> np.ndarray:
        return a * (1 - a)
        # dadz = super().dadz(a, z)
        # dadz[self.diag_idx(dadz)] = (a * (1 - a)).reshape(-1)
        # return dadz



class Softmax(Activation):

    def __call__(self, z: np.ndarray) -> np.ndarray:
        zexp = np.exp(z - z.min())
        zsum = zexp.sum(axis=tuple(range(1, z.ndim)))
        return zexp / zsum.reshape(len(zexp), *([1] * (z.ndim-1)))     
    

    def dadz(self, a: np.ndarray, z: np.ndarray) -> np.ndarray:
        #TODO: Computer multivariable gradient, diagonals: a_i (1 - a_i), off:
        # a_i (- a_k). Each element is d(a_i)/d(z_k) for i,k in activation units.
        return a * (1 - a)
