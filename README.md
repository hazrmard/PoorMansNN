# PoorMansNN

A simple implementation of a feed-forward neural network with minimal dependencies.

![Logo](/PoorMansNN.png)

The package defines a standard neural network `NN` class and several modules:

* `act`: Activation functions and their derivatives. Defines the `Activation` class.
* `loss`: Error functions and their derivatives. Defines the `Loss` class.
* `opt`: Optimization functions. Defines the `Optimizer` class.
* `layers`: Densly connected *n*-dimensional layers. Defines the `Layer` class.

Currently, implemented optimizers are:
* Momentum
* Standard Gradient Descent

Activations available:
* *tanh*
* *sigmoid*
* *ReLU*
* *LeakyReLU*
* *Linear*

Error measures:
* Squared error

All modules are extensively commented and provide function and class signatures
for modifications.

The notebook *Learning logic gates.ipynb* contains examples of usage.

## Usage example
Implementing a simple `XOR` gate:

```
Input   Hidden  Output
[]------[]--\
    X        >---[]
[]------[]--/
```

```python
import numpy as np
from poormansnn import NN, Layer, loss, act, opt

# Define hyperparameters and architecture
batchsize = 40
epochs = 500
layers = [Layer(shape=(2,), prevshape=(2,), act.Tanh()),
          Layer(shape=(1,), prevshape=(2,), act.Tanh())]
error = loss.SquaredLoss()
rate = 1.2
optimizer = opt.Optimizer(rate)

# Construct the network
n = NN(layers, error=error, optimizer=optimizer)

# Specify training data and labels. Each instance has the same dimension as
# the network's input layer. Each label has the same dimension as the network's
# output layer.
X = np.array([[0,0],
             [0,1],
             [1,0],
             [1,1]])
Y = np.array([[0],
              [1],
              [1],
              [0]])

# Train the network
errors, _ = n.train(X, Y, batchsize, epochs, train=(Y, X))
print(np.round(n.predict(X)))

```

![Illustration of XOR gate learning.](/illustration.png)