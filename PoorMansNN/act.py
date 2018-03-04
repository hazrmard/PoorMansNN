"""
Contains activation functions for units and derivatives of activations w.r.t
their arguments. Each function accepts a numpy array N x D of N, D-dimensional
input instances. Returns an array of thte same shape.
"""
import numpy as np



class Activation:
    """
    The activation function and its derivative. An activation is defined by:
    * **a**: The output of the activation function,
    * **z**: The input to the activation function.

    Mathematically:
    ```
    A = act(z)
    ```
    """

    def act(self, z: np.ndarray) -> np.ndarray:
        pass


    def dadz(self, a: np.ndarray, z: np.ndarray) -> np.ndarray:
        pass



class Linear(Activation):

    def act(self, z: np.ndarray) -> np.ndarray:
        return z

    def dadz(self, a: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.ones_like(z)



class Tanh(Activation):

    def act(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)


    def dadz(self, a: np.ndarray, z: np.ndarray) -> np.ndarray:
        return 1 - np.square(a)



class Relu(Activation):
        
    def act(self, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, z, 0)


    def dadz(self, a: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, 1., 0)




class Leaky_relu(Activation):
        
    def act(self, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, z, 1e-2*z)


    def dadz(self, a: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, 1., 1e-2)




class Sigmoid(Activation):
        
    def act(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))


    def dadz(self, a: np.ndarray, z: np.ndarray) -> np.ndarray:
        return a * (1 - a)
