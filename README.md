# PoorMansNN

A simple implementation of a feed-forward neural network with minimal dependencies.

The package defines a standard neural network `NN` class and several modules:

* `act`: Activation functions and their derivatives.
* `err`: Error functions and their derivatives.
* `opt`: Optimization functions.

Currently, momentum and standard gradient descent optimizers are implemented
with *tanh*, *sigmoid*, *ReLU*, *LeakyReLU*, and *Linear* activations. Squared
error is used by default.

All modules are extensively commented and provide function and class signatures
for modifications.

The notebook *Learning logic gates.ipynb* contains examples of usage.

![Illustration of XOR gate learning.](/illustration.png)