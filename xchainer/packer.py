# -*- coding: utf_8 -*-
from chainer import Variable


class NNpacker():
    def __init__(self, layers, **params):
        # NNを構成するレイヤーの関数を設定。layersにdictで名前をつけて管理する
        # 最終的にFunctionSetsで集約管理する
        # データのエントリポイントの名前をリストで設定する。リンクだけで、エントリポイントを持たない場合は空リストで省略可能
        entryPoints = params['entryPoints'] if 'entryPoints' in params else []
        children = params['children'] if 'children' in params else {}

        self.entryPoints = entryPoints
        self.layers = layers
        self.children = children

    def getFunctions(self):
        if len(self.children) > 0:
            # もしchildnnsが0じゃないなら、すべてのchildnnについてfunctionsを取り出して、自分のfunctionsと結合する
            chldList = self.children.values()
            chldFuncs = [child.getFunctions() for child in chldList]
            funcs = [self.layers] + chldFuncs
            return reduce(lambda acc, func: dict(acc, **func), funcs)
        else:
            return self.layers

    def execute(self, datasets, train=True):
        self.entryDatas = self.insource(datasets)
        self.childrenOutputs = self.outsource(datasets, train)
        return self.network(self.entryDatas, self.childrenOutputs, train)

    def insource(self, datasets):
        # 最上位のNNにデータ(dict型)を渡したら、下位のNNに振り分ける
        # なおかつ、連結された各NNへの入力を関数の呼び出し連鎖で表現する
        entries = [{ep: Variable(datasets[ep])} for ep in self.entryPoints]
        return reduce(lambda acc, entry: dict(acc, **entry), [{}] + entries)

    def outsource(self, datasets, train):
        chld = self.children.iteritems()
        chldOut = [{chn: ch.execute(datasets, train)} for chn, ch in chld]
        return reduce(lambda acc, output: dict(acc, **output), [{}] + chldOut)

    def network(self, e_data, c_data, train):
        # NNpacker一つ一つの中にデータを入ってきたときの動きを書く
        raise NotImplementedError("`network` method is not implemented.")

        # ex) self.layers.items() = [l1, l2, l3] のとき
        # h1 = F.relu(self.layers.l1(data))
        # h2 = F.relu(self.layers.l2(h1))
        # output = F.relu(self.layers.l3(h2))
        # return output
