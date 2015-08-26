from xchainer.packer import NNpacker
import unittest

import chainer.functions as F


class TestNNP(NNpacker):
    def __init__(self):
        layers = {
            'l1': F.Linear(784, 100),
            'l2': F.Linear(100, 100)
        }
        eps = ['target']
        children = {'child': ChildNNP()}
        NNpacker.__init__(self, layers, entryPoints=eps, children=children)

    def network(self, e_data, c_data, train):
        return e_data['target'].data + " AND " + c_data['child'].data


class ChildNNP(NNpacker):
    def __init__(self):
        layers = {}
        entryPoints = ['chtarget']
        NNpacker.__init__(self, layers, entryPoints=entryPoints)

    def network(self, e_data, __nochild__, train):
        return e_data['chtarget']


class NNpackerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.nnp = TestNNP()

    def test_getFunctions(self):
        print "===Test `getFunctions` method==="
        funcs = self.nnp.getFunctions()
        self.assertTrue("l1" in funcs)
        self.assertTrue("l2" in funcs)
        print "...done\n"

    def test_insource(self):
        print "===Test `insource` method==="
        datasets = {
            'target': "this is target data",
            'other': "this is not target data"
        }

        entryDatas = self.nnp.insource(datasets)
        self.assertTrue('target' in entryDatas)
        self.assertTrue('other' not in entryDatas)
        print "...done\n"

    def test_outsource(self):
        print "===Test `outsource` method==="
        datasets = {"chtarget": "this is target data of child"}
        childrenOutput = self.nnp.outsource(datasets, False)
        self.assertTrue("child" in childrenOutput)
        print "...done\n"

    def test_execute(self):
        print "===Test `execute` method==="
        datasets = {
            "target": "This is target data.",
            "chtarget": "This is target data for child."
        }
        result = self.nnp.execute(datasets)
        expected = datasets["target"] + " AND " + datasets["chtarget"]
        self.assertEqual(result, expected)
        print "...done\n"

if __name__ == '__main__':
    unittest.main()
