import unittest
from typing import List
import numpy as np
from poormansnn import NN, Layer, SoftmaxLayer, act, loss, opt


class TestActivation(unittest.TestCase):

    def setUp(self):
        self.Z: List[np.ndarray] = []
        self.dim_size = 10      # max size of any dimension
        self.max_dims = 4       # maximum dimensionality of inputs
        self.delta = 1e-3
        for i in range(1, self.max_dims):
            for j in range(1, self.max_dims):
                dim = np.random.choice(np.arange(1, self.dim_size + 1), j)
                batchsize = np.random.randint(1, self.dim_size+1)
                z = 1 + np.random.rand(batchsize, *dim)
                self.Z.append(z)
    


    def gradient_check(self, activation: act.Activation):
        # Comparing numerical and analytical gradients
        for z in self.Z:
            dz = np.random.rand(*z.shape) * self.delta
            zplus = z + dz / 2
            zminus = z - dz / 2

            aplus = activation(zplus)
            aminus = activation(zminus)
            a = activation(z)

            da = aplus - aminus
            dadz_n = da / dz
            dadz_a = activation.dadz(a, z)
            self.assertLessEqual(np.abs(dadz_a - dadz_n).sum(), self.delta)




    def test_linear(self):
        self.gradient_check(act.Linear())



    def test_sigmoid(self):
        self.gradient_check(act.Sigmoid())



    def test_tanh(self):
        self.gradient_check(act.Tanh())



    def test_relu(self):
        self.gradient_check(act.Relu())



    def test_leaky_relu(self):
        self.gradient_check(act.Leaky_relu())
    


    def test_softmax(self):
        # TODO: finish
        activation = act.Softmax()
        for z in self.Z:
            idx = []
            for i in range(1, z.ndim):
                index = np.random.randint(z.shape[i], size=z.shape[0])
                idx.append(index)
            idx = [np.arange(z.shape[0], dtype=int)] + idx
            mask = np.ones(z.shape, dtype=bool)
            mask[idx] = 0

            dz = np.random.rand(*z.shape) * self.delta
            zplus = z + dz / 2
            zminus = z - dz / 2

            aplus = activation(zplus)
            aplus[mask] = 0
            aminus = activation(zminus)
            aminus[mask] = 0
            a = activation(z)
            a[mask] = 0

            da = aplus - aminus
            dadz_n = da / dz
            # print(dadz_n)
            dadz_a = activation.dadz(a, z)
            # print(dadz_a)
            # self.assertLessEqual(np.abs(dadz_a - dadz_n).sum(), self.delta)



class TestLayer(unittest.TestCase):


    @staticmethod
    def einsumProduct(x: np.ndarray, w: np.ndarray) -> np.ndarray:
        # An alternative approach for tensor multiplication using einstein
        # notation. Used for verifying FeedForward computations. Hardcoded for
        # multiplying N x dim(x) input batch with dim(x) x dim(a) weight matrix.
        return np.einsum(x, list(range(0, len(x.shape))),
                w, list(range(1, len(w.shape)+1)),
                [0, *list(range(len(x.shape), len(w.shape)+1))])


    def setUp(self):
        self.dim_size = 10      # max size of any dimension
        self.max_dims = 4       # maximum dimensionality of layers
        self.layers = []
        self.delta = 1e-2
        for i in range(1, self.max_dims):
            for j in range(1, self.max_dims):
                batchsize = np.random.randint(1, self.dim_size+1)
                dim = np.random.choice(np.arange(1, self.dim_size + 1), j)
                indim = np.random.choice(np.arange(1, self.dim_size + 1), i)
                l = Layer(dim, indim, act.Linear())
                l.x = np.random.rand(batchsize, *indim)
                l.w = np.random.rand(*l.w.shape)
                self.layers.append(l)




    def test_feedforward(self):
        # iterate over different combination of dimensions for layer input and
        # outputs
        for l in self.layers:
            y = l.feedforward(l.x)
            # check if output dimensions are as expected
            self.assertTrue(y.shape==(len(l.x), *l.shape))
            # compute output using einstein notation to verify tensordot
            # approach
            yein = l.act(self.einsumProduct(l.x, l.w))
            self.assertAlmostEqual((y-yein).sum(), 0., 7)



    def test_gradient(self):
        # iterate over different combination of dimensions for layer input and
        # outputs
        for l in self.layers:
            dEda = np.array([1])
            out = l.feedforward(l.x)
            dEdz = l.dEdz(dEda)
            dEdw_a = l.dEdw(dEdz)
            dEdb = l.dEdb(dEdz)
            dEdx = l.dEdx(dEdz)
            self.assertTrue(dEdz.shape==l.a.shape)
            self.assertTrue(dEdw_a.shape==l.w.shape)
            self.assertTrue(dEdb.shape==l.b.shape)
            self.assertTrue(dEdx.shape==(len(l.x), *l.prevshape))
            # calculate numerical gradient for verification
            dw = np.random.rand(*l.w.shape) * self.delta
            w = l.w.copy()
            l.w = w + dw / 2
            outplus = l.feedforward(l.x)
            zplus = l.z.copy()
            l.w = w - dw / 2
            outminus = l.feedforward(l.x)
            zminus = l.z.copy()
            da = outplus - outminus
            dz = zplus - zminus
            dEdw_n = np.tensordot(l.x, dEda * (da / dz), axes=[[0], [0]]) / len(l.x)
            self.assertLessEqual(np.abs((dEdw_a - dEdw_n)).sum(), self.delta)



class TestNN(unittest.TestCase):

    def setUp(self):
        self.num_layers = 5
        self.dim_choices = [1, 2, 3]
        self.dim_size = 10
        self.batchsize = 10
        layers = []
        indim = np.random.choice(np.arange(1, self.dim_size + 1), np.random.choice(self.dim_choices))
        for i in range(self.num_layers - 1):
            dim = np.random.choice(np.arange(1, self.dim_size + 1), np.random.choice(self.dim_choices))
            l = Layer(dim, indim, act.Sigmoid())
            l.a = np.random.rand(self.batchsize, *dim)
            l.z = np.random.rand(self.batchsize, *dim)
            l.x = np.random.rand(self.batchsize, *indim)
            layers.append(l)
            indim = dim
        dim = np.random.choice(np.arange(1, self.dim_size + 1), np.random.choice(self.dim_choices))
        l = SoftmaxLayer(dim, indim)
        l.a = np.random.rand(self.batchsize, *dim)
        l.z = np.random.rand(self.batchsize, *dim)
        l.x = np.random.rand(self.batchsize, *indim)
        layers.append(l)
        self.x = np.random.rand(self.batchsize, *layers[0].prevshape)
        self.y = np.zeros((self.batchsize, *layers[-1].shape))
        idx = []
        for i in range(0, len(l.shape)):
            index = np.random.randint(l.shape[i], size=self.batchsize)
            idx.append(index)
        idx = [np.arange(self.batchsize, dtype=int)] + idx
        self.y[idx] = 1
        self.nn = NN(layers, loss.CrossEntropyLoss(), opt.Optimizer(0.1))



    def test_prediction(self):
        y = self.nn.predict(self.x)
        self.assertTrue(y.shape==self.nn.layers[-1].a.shape)



    def test_backpropagation(self):
        pred = np.random.rand(self.batchsize, *self.nn.layers[-1].shape)
        self.nn.backpropagate(self.y, pred)



    def test_training(self):
        err, _ = self.nn.train(self.x, self.y, self.batchsize // 2, 5, test=(self.y, self.x))




if __name__ == '__main__':
    unittest.main()