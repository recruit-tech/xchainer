import os
import sys
sys.path.append(os.getcwd())

from xchainer.manager import NNmanager
from xchainer.packer import NNpacker

import numpy as np
from chainer import FunctionSet, optimizers
import chainer.functions as F
from sklearn.base import ClassifierMixin
from sklearn import cross_validation
from sklearn.datasets import fetch_mldata


class Union(NNpacker):
    def __init__(self):
        layers = {
            'union_l1': F.Linear(100, 50),
            'union_l2': F.Linear(50, 50),
            'union_l3': F.Linear(50, 10)
        }
        children = {'upper': Upper(), 'lower': Lower()}
        NNpacker.__init__(self, layers, children=children)

    def network(self, __noentry__, children, train):
        upper = children['upper']
        lower = children['lower']
        data = F.concat((upper, lower))
        h1 = F.relu(self.layers['union_l1'](data))
        h2 = F.relu(self.layers['union_l2'](h1))
        output = F.relu(self.layers['union_l3'](h2))
        return output


class Upper(NNpacker):
    def __init__(self):
        layers = {
            'upper_l1': F.Linear(392, 100),
            'upper_l2': F.Linear(100, 100),
            'upper_l3': F.Linear(100, 50)
        }
        NNpacker.__init__(self, layers, entryPoints=['upper'])

    def network(self, entry, __nochild__, train):
        data = entry['upper']
        h1 = F.relu(self.layers['upper_l1'](data))
        h2 = F.relu(self.layers['upper_l2'](h1))
        output = F.relu(self.layers['upper_l3'](h2))
        return output


class Lower(NNpacker):
    def __init__(self):
        layers = {
            'lower_l1': F.Linear(392, 100),
            'lower_l2': F.Linear(100, 100),
            'lower_l3': F.Linear(100, 50)
        }
        NNpacker.__init__(self, layers, entryPoints=['lower'])

    def network(self, entry, __nochild__, train):
        data = entry['lower']
        h1 = F.relu(self.layers['lower_l1'](data))
        h2 = F.relu(self.layers['lower_l2'](h1))
        output = F.relu(self.layers['lower_l3'](h2))
        return output


class MnistComplex(NNmanager, ClassifierMixin):
    def __init__(self, nnpacker, logging=False):
        self.nnpacker = nnpacker
        model = FunctionSet(**nnpacker.getFunctions())
        optimizer = optimizers.SGD()
        lossFunction = F.softmax_cross_entropy
        params = {'epoch': 20, 'batchsize': 100, 'logging': logging}
        NNmanager.__init__(self, model, optimizer, lossFunction, **params)

    def trimOutput(self, output):
        y_trimed = output.data.argmax(axis=1)
        return np.array(y_trimed, dtype=np.int32)

    def forward(self, x_batch, train):
        x_data = {'upper': x_batch[:, 0:392], 'lower': x_batch[:, 392:784]}
        return self.nnpacker.execute(x_data)

mnist = fetch_mldata('MNIST original')
x_all = mnist['data'].astype(np.float32) / 255
y_all = mnist['target'].astype(np.int32)

union = Union()
mc = MnistComplex(union, logging=True)
score = cross_validation.cross_val_score(mc, x_all, y_all, cv=2)
print score.mean()
