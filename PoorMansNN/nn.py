"""
Implements a feed forward neural network.
"""

from typing import List, Dict, Tuple, Callable
from copy import deepcopy
import numpy as np
from . import opt
from . import loss
from . import Layer

class NN:
    """
    A simple feed-forward neural network.

    Args:
    * layers (List[int]): A list of Layer instances for hidden and output layers.
    * error (loss.Loss): The error measure function and derivative.
    * optimizer (opt.Optimizer): Changes weights in response to error derivatives.

    Attributes:
    * L (int): Number of layers in network (hidden, output).
    * layers (List[Layer]): A list of number of units in each layer.
    * error (loss.Loss): Returns the error measure.
    * optimizer (opt.Optimizer): An instantiated Optimizer class.
    """

    def __init__(self, layers: List[Layer], error: loss.Loss,
                 optimizer: opt.Optimizer):

        self.L = len(layers)
        self.layers = layers
        self.error = error
        # Since optimizers may have 'memory' (e.g momentum which remembers the
        # last update value), each layer's weight and bias arrays are assigned
        # their copy of optimizer.
        self.weight_optimizers = [deepcopy(optimizer) for _ in range(self.L)]
        self.bias_optimizers = deepcopy(self.weight_optimizers)



    def feedforward(self, x: np.ndarray) -> List[np.ndarray]:
        """
        Propagates a batch of n, d-dimensional inputs forward through L-layers.

        Args:
        * x (np.ndarray): a n x d array of n, d-dimensional input values

        Returns:
        * A list of L-activations for each layer (hidden, output). Each
        element is a n x units array of activation values for each unit in a
        layer for each input instance in x.
        """
        activations = []
        for l in self.layers:
            x = l.feedforward(x)
            activations.append(x)
        return activations    



    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts labels from inputs.

        Args:
        * x (np.ndarray): a n x d array of n, d-dimensional input values

        Returns:
        * A n x d array of values representing activations of the last layer.
        """
        for l in self.layers:
            x = l.feedforward(x)
        return x



    def backpropagate(self, y: np.ndarray, pred: np.ndarray):
        """
        Given the state of network units and the expected output, propagates
        error corrections backwards and changes weights on each layer.

        Args:
        * activations (List[np.ndarray]): A list of activations for each layer
        (input, hidden, output). Each element is a n x units array of activation
        values for each unit in a layer for each input instance in the minibatch.
        * y (np.ndarray): An array of n, d-dimensional vectors representing the
        true label/output layer values for the n instances.
        """
        # Initial error w.r.t predicted output, of shape n x output units
        dEda = self.error.dEdy(y, pred)
        # Indices: i = L-2, L-3, ..., 0.
        for i in range(self.L-1, -1, -1):
            # Error w.r.t logit of current layer for n instances in minibatch
            # Shape: N x dim(layer)
            dEdz = self.layers[i].dEdz(dEda)
            # Error w.r.t activations of previous layer (inputs of current layer)
            # Shape: N x dim(prev layer)
            dEda = self.layers[i].dEdx(dEdz)
            # Mean error w.r.t weights of current layer
            # Shape: inputs x units = inputs x n * n x units
            dEdw = self.layers[i].dEdw(dEdz)
            # Mean error w.r.t biases of current layer
            # Shape: dim(layer)
            dEdb = self.layers[i].dEdb(dEdz)
            # Gradient descent on weights and biases
            # self.weights[i] += self.weight_optimizers[i](dEdw)
            # self.biases[i] += self.bias_optimizers[i](dEdb)
            self.layers[i].w += self.weight_optimizers[i](dEdw)
            self.layers[i].b += self.bias_optimizers[i](dEdb)



    def train(self, X: np.ndarray, Y: np.ndarray, batchsize: int, epochs: int,
        weighthist=False, **test) -> Tuple[Dict[str, List[float]], List[np.ndarray]]:
        """
        Given N training inputs (of dimension==input layer) and their labels
        (of dimensions==output layer), backpropagates errors to find optimal
        weights.

        Args:
        * X (np.ndarray): A N x d array of N, d-dimensional input instances.
        * Y (np.ndarray): a N x e array of N, e-dimensional output labels.
        * batchsize (int): Training sample size.
        * epochs (int): Number of times to go over training data.
        * weighthist (bool): If true, stores weights at end of each epoch.
        * any number of name=(labels, inputs) keyword arguments for instances to
        record error on for each epoch. Labels and inputs same type as Y, X.

        Returns:  
        A tuple of:
            * dict of name=List[float] containing error histories for each epoch
            for the test labels and inputs provided.
            * list of epochs x inputs x units arrays of weight and bias histories
            for each layer. If weighthist=False, list is empty. If there are L
            hidden and output layers, the first L elements are weights, the last
            L elements are biases.
        """
        errors = {t:[] for t in test}
        if weighthist:
            whist = [np.zeros((epochs, *l.w.shape)) for l in self.layers]
            bhist = [np.zeros((epochs, *l.b.shape)) for l in self.layers]
        else:
            whist = []
            bhist = []
        N = len(X)
        batches = int(np.ceil(N / batchsize))
        for i in range(epochs):
            # shuffling order of training data by index (indices may repeat)
            indices = np.random.randint(0, N, N)
            for j in range(batches):
                x = X[indices[j*batchsize:(j+1)*batchsize]]
                y = Y[indices[j*batchsize:(j+1)*batchsize]]
                pred = self.predict(x)
                self.backpropagate(y, pred)
            for t, errhist in errors.items():
                errhist.append(self.error(test[t][0], self.predict(test[t][1])))
            if weighthist:
                for l in range(self.L):
                    whist[l][i] = self.layers[l].w
                    bhist[l][i] = self.layers[l].b
        return errors, whist + bhist
