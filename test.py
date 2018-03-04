import unittest
import numpy as np
from poormansnn import NN, Layer, act, loss, opt


class TestActivation(unittest.TestCase):

    def setUp(self):
        pass
    


    def test_linear(self):
        pass
    


    def test_sigmoid(self):
        pass
    


    def test_tanh(self):
        pass
    


    def test_relu(self):
        pass
    


    def test_leaky_relu(self):
        pass



class TestLayer(unittest.TestCase):

    def setUp(self):
        self.batchsize = 10
        self.dim_size = 10
    


    def test_feedforward(self):
        for i in range(1, 4):
            for j in range(1, 4):
                dim = np.random.choice(np.arange(1, self.dim_size), j)
                indim = np.random.choice(np.arange(1, self.dim_size), i)
                l = Layer(dim, indim, act.Linear())
                x = np.random.rand(self.batchsize, *indim)
                y = l.feedforward(x)
                self.assertTrue(y.shape==(self.batchsize, *dim))
    


    def test_gradients(self):
        for i in range(1, 4):
            for j in range(1, 4):
                dim = np.random.choice(np.arange(1, self.dim_size), j)
                indim = np.random.choice(np.arange(1, self.dim_size), i)
                l = Layer(dim, indim, act.Linear())
                dEda = np.random.rand(self.batchsize, *dim)
                l.z = np.random.rand(self.batchsize, *dim)
                l.a = l.act.act(l.z)
                l.x = np.random.rand(self.batchsize, *indim)
                dEdz = l.dEdz(dEda)
                dEdw = l.dEdw(dEdz)
                dEdb = l.dEdb(dEdz)
                dEdx = l.dEdx(dEdz)
                self.assertTrue(dEdz.shape==dEda.shape)
                self.assertTrue(dEdw.shape==l.w.shape)
                self.assertTrue(dEdb.shape==l.b.shape)
                self.assertTrue(dEdx.shape==(self.batchsize, *indim))



class TestNN(unittest.TestCase):

    def setUp(self):
        self.num_layers = 5
        self.dim_choices = [1, 2, 3]
        self.dim_size = 10
        self.batchsize = 10
        layers = []
        indim = np.random.choice(np.arange(1, self.dim_size), np.random.choice(self.dim_choices))
        for i in range(self.num_layers):
            dim = np.random.choice(np.arange(1, self.dim_size), np.random.choice(self.dim_choices))
            l = Layer(dim, indim, act.Linear())
            l.a = np.random.rand(self.batchsize, *dim)
            l.z = np.random.rand(self.batchsize, *dim)
            l.x = np.random.rand(self.batchsize, *indim)
            layers.append(l)
            indim = dim
        self.nn = NN(layers, loss.SquaredLoss(), opt.Optimizer(0.1))
        self.x = np.random.rand(self.batchsize, *layers[0].prevshape)
        self.y = np.random.rand(self.batchsize, *layers[-1].shape)



    def test_prediction(self):
        y = self.nn.predict(self.x)
        self.assertTrue(y.shape==self.nn.layers[-1].a.shape)
    


    def test_backpropagation(self):
        pred = np.random.rand(self.batchsize, *self.nn.layers[-1].shape)
        self.nn.backpropagate(self.y, pred)



    def test_training(self):
        err = self.nn.train(self.x, self.y, self.batchsize // 2, 5, test=(self.y, self.x))




if __name__ == '__main__':
    unittest.main()