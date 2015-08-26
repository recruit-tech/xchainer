from xchainer.manager import NNmanager
import unittest
import sys

import numpy as np
from chainer import FunctionSet, Variable, optimizers
import chainer.functions as F

from sklearn.base import ClassifierMixin
from sklearn import cross_validation

from sklearn.datasets import fetch_mldata


class TestNNM(NNmanager, ClassifierMixin):
    def __init__(self, logging=False):
        model = FunctionSet(
            l1=F.Linear(784, 100),
            l2=F.Linear(100, 100),
            l3=F.Linear(100,  10)
        )

        optimizer = optimizers.SGD()
        lossFunction = F.softmax_cross_entropy
        params = {'epoch': 5, 'batchsize': 1000, 'logging': logging}
        NNmanager.__init__(self, model, optimizer, lossFunction, **params)

    def trimOutput(self, output):
        y_trimed = output.data.argmax(axis=1)
        return np.array(y_trimed, dtype=np.int32)

    def forward(self, x_batch, train):
        x = Variable(x_batch)
        h1 = F.relu(self.model.l1(x))
        h2 = F.relu(self.model.l2(h1))
        output = F.relu(self.model.l3(h2))
        return output


class NNmanagerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        try:
            print "Loading MNIST data for test. This could take a while..."
            mnist = fetch_mldata('MNIST original')
            self.x_all = mnist['data'].astype(np.float32) / 255
            self.y_all = mnist['target'].astype(np.int32)
            print "...done\n"
        except Exception:
            print "*** Error occured in loading MNIST data ***"
            print "It could be caused by old MNIST data in your machine."
            print "Tips: rm ~/scikit_learn_data/mldata/mnist-original.mat"
            sys.exit()

        self.nnm = TestNNM()
        indexes = np.random.permutation(len(self.x_all))
        self.trainIndexes = indexes[0:len(self.x_all)/2]
        self.testIndexes = indexes[len(self.x_all)/2:]
        self.setup = True

    def test_fit(self):
        print "===Test `fit` method==="
        print "This could take a while..."
        x_train = self.x_all[self.trainIndexes]
        y_train = self.y_all[self.trainIndexes]
        self.nnm.fit(x_train, y_train)
        self.assertTrue(True)
        print "...done\n"

    def test_predict(self):
        # y_predict should have same shape as y_test
        print "===Test `predict` method==="
        print "This could take a while..."
        x_test = self.x_all[self.testIndexes]
        y_test = self.y_all[self.testIndexes]
        y_predict = self.nnm.predict(x_test)
        self.assertEqual(y_predict.shape, y_test.shape)
        print "...done\n"

    def test_withsk_crossval(self):
        print "===Test learning with `cross_val_score` of sklearn==="
        print "logging learning process below..."
        nnm = TestNNM(logging=True)
        x = self.x_all
        y = self.y_all
        cvs = cross_validation.cross_val_score(nnm, x, y, cv=2)
        print cvs
        self.assertTrue(cvs is not None)
        print "...complete\n"


if __name__ == '__main__':
    unittest.main()
