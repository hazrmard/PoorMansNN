from typing import List, Dict, Tuple, Callable
from copy import deepcopy
import numpy as np
import opt

class NN:
    """
    A simple feed-forward neural network.

    Args:
    * layers (List[int]): A list of units in each layer for input, hidden, and
    output layers (for a total of L layers).
    * activations (List[Callable]): A list of activation functions for each layer
    after the input layer.
    * dactivations (List[Callable]): A list of activation derivative functions for
    each layer after the input layer i.e. d{activation(l)} / dl.
    * error (Callable): The error measure function.
    * derror (Callable): The error derivative i.e. d{Error(y, ypred)} / d ypred.
    * optimizer (Callable): A function/callable that takes the array of error
    derivatives w.r.t weights and returns a similar array containing weight
    changes for the update.

    Attributes:
    * L (int): Number of layers in network (input, hidden, output).
    * layers (List[int]): A list of number of units in each layer.
    * weights (List[np.ndarray]): A list of L-1, n x u arrays for each of hidden]
    and output layers. The ith array contains weights for n units in ith to u
    units in (i+1)th layers.
    * biases (List[np.ndarray]): A list of L-1 biase arrays for each of hidden
    and output layers. The ith array contains a 1 x u array of biases for the u
    units in the (i+1)th layer.
    * activate (List[func]): A list of L-1 functions for each of hidden and output
    laters. The ith function accepts an n x u array of n weighted sums for each of
    the u units in (i+1)th layer. Each function returns a similarly sized array
    containing activations.
    * dadl (List[func]): A list of L-1 functions of the same type as activate.
    Each function returns the derivative of the activation w.r.t its argument.
    * dEdy (func): A function that returns the derivative of the error w.r.t
    prediction. Takes a tuple of n x output-units arrays of labels for n
    instances in a minibatch, and a same shaped array of predicted labels. Returns
    a similarly shaped array.
    * error (func): Returns the error measure. Takes same inputs as dEdy() and
    returns a float.
    * optimizer (Callable): Takes a N x D array of error derivatives w.r.t
    weights for each N (inputs) x D (units) weights for a single layer. Returns
    a similar array containing weight updates.
    """

    def __init__(self, layers: List[int],
                 activations: List[Callable[[np.ndarray], np.ndarray]],
                 dactivations: List[Callable[[np.ndarray], np.ndarray]],
                 error: Callable[[np.ndarray, np.ndarray], float],
                 derror: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 optimizer: opt.Optimizer):

        self.L = len(layers)
        self.layers = layers
        # List of length len(layers)-1 weight arrays,
        # weights[i] is a inputs x units array of weights from layer i to i+1, 
        self.weights = []
        # List of length len(layers)-1 bias arrays,
        # biases[i] is a 1 x units array for layer i
        self.biases = []
        # A list of non-linearity functions for the weighted sum of inputs (logit) for
        # each unit for hidden and output layers. Takes a n-inputs x u-units array.
        # Returns the same. Length: L-1
        self.activate = activations
        # A list of derivatives of each layer's activation w.r.t the logit.
        # Takes n-instances x u-units array, returns the same. Length: L-1
        self.dadl = dactivations
        # Derivative of data error w.r.t predicted output. Takes a tuple of
        # n x output units array true labels for n instances in a minibatch, and
        # a same shaped array of predicted labels.
        self.dEdy = derror
        # The error function. Takes similar inputs as dEdy(), returns a float.
        self.error = error
        # Since optimizers may have 'memory' (e.g momentum which remembers the
        # last update value), each layer's weight and bias arrays are assigned
        # their copy of optimizer.
        self.weight_optimizers = [deepcopy(optimizer) for _ in range(self.L-1)]
        self.bias_optimizers = deepcopy(self.weight_optimizers)

        for i in range(1, len(layers)):
            inputs = layers[i-1]
            units = layers[i]
            self.biases.append(np.zeros((1, units)))
            # https://stats.stackexchange.com/q/47590
            self.weights.append(np.random.rand(inputs, units) * 2 / np.sqrt(inputs) - np.sqrt(inputs))
    


    def feedforward(self, x: np.ndarray) -> List[np.ndarray]:
        """
        Propagates a batch of n, d-dimensional inputs forward through L-layers.

        Args:
        * x (np.ndarray): a n x d array of n, d-dimensional input values

        Returns:
        * A list of L-activations for each layer (input, hidden, output). Each
        element is a n x units array of activation values for each unit in a
        layer for each input instance in x.
        """
        activations = [x]
        for i in range(self.L-1):
            # N x units = N x inputs * inputs x units + 1x units
            # units become inputs for next layer
            logit = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            activations.append(self.activate[i](logit))
        return activations
    


    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts labels from inputs.

        Args:
        * x (np.ndarray): a n x d array of n, d-dimensional input values

        Returns:
        * A n x d array of values representing activations of the last layer.
        """
        return self.feedforward(x)[-1]



    def backpropagate(self, activations: List[np.ndarray], y: np.ndarray):
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
        dEda = self.dEdy(y, activations[-1])
        # Indices: i = L-2, L-3, ..., 0.
        for i in range(self.L-2, -1, -1):
            # Error w.r.t logit of current layer for n instances in minibatch
            # Shape: n x units (curr layer)
            dEdl = dEda * self.dadl[i](activations[i+1])
            # Error w.r.t activations of previous layer (inputs of current layer)
            # Shape: n x inputs (prev layer units) = n x units * units x inputs
            dEda = np.dot(dEdl, self.weights[i].T)
            # Mean error w.r.t weights of current layer
            # Shape: inputs x units = inputs x n * n x units
            dEdw = np.dot(activations[i].T, dEdl) / len(y)
            # Mean error w.r.t biases of current layer
            # Shape: 1 x units
            dEdb = np.mean(dEdl, 0)
            # Gradient descent on weights and biases
            self.weights[i] += self.weight_optimizers[i](dEdw)
            self.biases[i] += self.bias_optimizers[i](dEdb)



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
            whist = [np.zeros((epochs, *w.shape)) for w in self.weights]
            bhist = [np.zeros((epochs, *b.shape)) for b in self.biases]
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
                activations = self.feedforward(x)
                self.backpropagate(activations, y)
            for t, errhist in errors.items():
                errhist.append(self.error(test[t][0], self.predict(test[t][1])))
            if weighthist:
                for w in range(len(self.weights)):
                    whist[w][i] = self.weights[w]
                    bhist[w][i] = self.biases[w]
        return errors, whist + bhist
