import os
import sys
sys.path.append(os.getcwd())

from xchainer.manager import NNmanager
import numpy as np
from chainer import FunctionSet, Variable, optimizers
import chainer.functions as F
from sklearn.base import ClassifierMixin
from sklearn import cross_validation
from sklearn.datasets import fetch_mldata


class MnistSimple(NNmanager, ClassifierMixin):
    def __init__(self, logging=False):
        model = FunctionSet(
            l1=F.Linear(784, 100),
            l2=F.Linear(100, 100),
            l3=F.Linear(100,  10)
        )

        optimizer = optimizers.SGD()
        lossFunction = F.softmax_cross_entropy
        params = {'epoch': 20, 'batchsize': 100, 'logging': logging}
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

mnist = fetch_mldata('MNIST original')
x_all = mnist['data'].astype(np.float32) / 255
y_all = mnist['target'].astype(np.int32)

ms = MnistSimple(logging=True)
score = cross_validation.cross_val_score(ms, x_all, y_all, cv=2)
print score.mean()
