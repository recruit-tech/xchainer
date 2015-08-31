#Overview
##About
Scikit-learnとのアダプターなどを提供するChainerの拡張モジュールです。
本モジュールの目的は、Chainerにおける学習プロセスの記述の簡略化及び評価・検定手段の拡充です。
Scikit-learnの評価・検定モジュールを利用するために、Scikit-learnの学習器としてChainerをラップしています。
Chainerの基本的な使い方につきましては、公式のチュートリアルをご参照ください。

* [Chainer Tutorial](http://docs.chainer.org/en/latest/tutorial/index.html)

##Coding style
本モジュールのコードは、python標準のPEP8に則って開発しています。

* [pep8 日本語ドキュメント](http://pep8-ja.readthedocs.org/ja/latest/)

##Quick Start
###Install
```shell
git clone https://github.com/recruit-tech/xchainer.git
cd xchainer
pip install -r requirements.txt
pip install .
```
###Test
```shell
$ python -m unittest discover -s tests
```

###Examples
```shell
$ python ./examples/mnist_simple.py
```

```shell
$ python ./examples/mnist_complex.py
```

#Documentation
##NNmanager
`NNmanager`は、学習プロセスのパラメータ化により、必要最低限の記述によるネットワークの定義を可能にします。
また、`NNmanager`はScikit-learnの学習器として拡張されているため、交差検定やAUC評価など、Scikit-learnから提供されている様々な評価・検定モジュールを利用することができます。


###Start with Example
`NNmanager`は学習器の枠組みを提供するインタフェースです。`NNmanager`を継承し、目的に応じて拡張することで、学習器を作ることができます。
継承の際必要になるのは、ネットワーク構造`model`、最適化関数`optimizer`、損失関数`lossFunction`の三つです。ここで、`model`は`chainer.FunctionSet`クラスのインスタンスで、ネットワークのパラメータを全てまとめて管理する役目を持ちます。`optimizer`は`chainer.optimizers`で提供される最適化関数、`lossFunction`は`chainer.functions`で提供される損失関数です。
詳しくは[chainerのリファレンスマニュアル](http://docs.chainer.org/en/latest/reference/index.html)をご参照ください。

* [chainer FunctionSet](http://chainer.readthedocs.org/en/latest/reference/core/function_set.html)
* [chainer optimizers](http://chainer.readthedocs.org/en/latest/reference/optimizers.html)
* [chainer functions](http://chainer.readthedocs.org/en/latest/reference/functions.html)

これらに加えて、オプションとして`params`を渡すことができます。`params`はdict型です。設定できる項目は、エポック数`epoch`、バッチサイズ`batchsize`、学習ログ表示フラグ`logging`です。
拡張の際に必要になるのは、`forward`メソッドと`trimOutput`メソッドの定義です。これにより、学習器を具体化します。

ここでは、例として手書き文字認識のデータを対象にしたネットワークをあげます。

```python
from xchainer import NNmanager
import numpy as np
from chainer import FunctionSet, Variable, optimizers
import chainer.functions as F
from sklearn.base import ClassifierMixin

# NNmanagerとClassifierMixinの継承
class TestNNM(NNmanager, ClassifierMixin):
    def __init__(self, logging=False):
        # ネットワーク構造の定義
        model = FunctionSet(
            l1=F.Linear(784, 100),
            l2=F.Linear(100, 100),
            l3=F.Linear(100,  10)
        )
        # 最適化手法の選択
        optimizer = optimizers.SGD()
        # 損失関数の選択
        lossFunction = F.softmax_cross_entropy
        # パラメータの設定
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
```

今回扱う手書き文字認識は、1~10までの10種類の数字を判別する10クラスの分類問題なので、`ClassifierMixin`を利用しています。回帰問題を対象とする場合には、`RegressorMixin`を使います。

* [Scikit-learn ClassifierMixin](http://scikit-learn.org/stable/modules/generated/sklearn.base.ClassifierMixin.html)
* [Scikit-learn RegressorMixin](http://scikit-learn.org/stable/modules/generated/sklearn.base.RegressorMixin.html)

####forward
NNmanagerでは順伝播`forward`メソッドを定義すれば、ニューラルネットワークの学習過程を構築できます。
ニューラルネットワークにおける学習は、ネットワーク構造に強く依存します。Chainerでは、基本的にネットワーク構造に即した定義が必要なのは順方向の伝播だけで、その他の過程は一般化することができます。
`forward`メソッドは、ネットワークの入力層への入力`x_batch`を受け取り、出力層からの出力`output`を返します。ここで、`output`は`chainer.Variable`クラスのインスタンスです。`train`はネットワークの学習フラグで、`fit`の際には`True`、`predict`の際には`False`が入ります。


```python
# 上略
    def forward(self, x_batch, **options):
        x = Variable(x_batch)
        h1 = F.relu(self.model.l1(x))
        h2 = F.relu(self.model.l2(h1))
        output = F.relu(self.model.l3(h2))
        return output
```

####trimOutput
`trimOutput`メソッドは、`forward`メソッドの結果である`output`を受け取り、ネットワークの出力値をラベル（被説明変数）と比較可能な形で取り出します。Scikit-learnの検定・評価モジュールを使う際には、`chainer.Variable`型のままでは扱えないためです。
`trimOutput`メソッドは、デフォルトで`output.data`を取り出して返すので、回帰問題の際にはメソッド・オーバーライドは必要ありません。今回は10クラスの分類問題であるため、10次元列ベクトルの出力値の中で最も大きな値を持つ行番号をラベル値として取得しています。

```python
# 上略
    def trimOutput(self, output):
        y_trimed = output.data.argmax(axis=1)
        return np.array(y_trimed, dtype=np.int32)
```

###Try Example
上記のサンプルコードは、`./examples/mnist_simple.py`で試すことができます

```shell
$ python ./examples/mnist_simple.py
```

##NNpacker
`NNpakcer`は、ネットワーク構造をカプセル化することにより、複雑な階層構造を持つネットワークの定義・操作を簡略化します。

###Start with Example
上述の`NNmanager`で用いた手書き文字認識のサンプルケースを改造し、少し変わったネットワークを作ることを考えます。
ここでは、手書き文字画像の上半分と下半分を別々に学習する場合を考えます。この場合、ネットワークは複数の小さなネットワークから構成されます。具体的には、上半分と下半分を受け取るネットワークが一つずつと、それらの結果を集約するネットワークが一つの合計三つのセグメントからなります。
このネットワークは、一つの親ノードと二つの子ノードという形で表現することができます。`NNpacker`は、ネットワーク構造をノード一つ一つに凝縮し、つなぎ合わせることができるようにします。

`NNmanager`同様、`NNpacker`も抽象クラスなので、継承する具体クラスを定義します。ここで、親ノードのクラスを`Union`、子ノードのクラスを`Upper`と`Lower`とします。
`Upper`と`Lower`は、それぞれ一つのデータ入力（画像の上半分と下半分）を受け付けるネットワークです。`NNpacker`では、データ入力を受け付ける端子を`entryPoint`と呼びます。
一方、`Union`は二つの子`Upper`と`Lower`を持つネットワークです。`NNpacker`では、ネットワークが持つ子は`children`で表されます。また`Union`は`entryPoint`を持たず、子である`Upper`と`Lower`の出力のみを扱います。

このネットワークを図示すると、以下のようになります。
![nnpacker](https://github.com/recruit-tech/xchainer/blob/master/images/nnpacker.png)

`NNpacker`では、親子関係とエントリーポイントにより、最上位の親ノードに位置するネットワークから連なる全てのネットワークを集約管理することができます。

継承に必要なのは、`layers`の設定と`network`メソッドの定義です。
`layers`はネットワークの階層ごとに名前をつけて管理するdict型オブジェクトです。
`network`メソッドは、エントリーポイントから入力されたり、子から渡されたりしたデータをネットワークに適用する処理を行います。

```python
from module.packer import NNpacker
import chainer.functions as F


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
        
    
class Union(NNpacker):
    def __init__(self):
        layers = {
            'union_l1': F.Linear(100, 50),
            'union_l2': F.Linear(50, 50),
            'union_l3': F.Linear(50, 10)
        }
        children={'upper': Upper(), 'lower': Lower()}
        NNpacker.__init__(self, layers, children=children)

    def network(self, __noentry__, childrenOutput, train):
        upper = childrenOutput['upper']
        lower = childrenOutput['lower']
        data = F.concat((upper, lower))
        h1 = F.relu(self.layers['union_l1'](data))
        h2 = F.relu(self.layers['union_l2'](h1))
        output = F.relu(self.layers['union_l3'](h2))
        return output
```

####execute
このように`NNpacker`を用いて定義した複合的なネットワークは、最上位ネットワーク構造の`execute`メソッドに入力データを渡すことで順伝播処理を走らせることができます。
今回は、`Union`のインスタンスを作り、`execute`メソッドを呼び出すことになります。このときに渡す入力データは、各エントリーポイントの名前をキーにして、データを格納したdict型のオブジェクトになります。

```python
# xはmnistのデータとします
union = Union()
x_data = {'upper': x[:, 0:392], 'lower': x[:, 392:784]}
union.execute(x_data)
```

####Work with NNmanager
`NNpacker`は、ネットワーク構造を切り出し、ラッピングしたものです。
`NNpacker`クラスのインスタンスが持つ全ネットワーク構造は`getFunctions`で取得することができます。

```python
from module.manager import NNmanager
import numpy as np
from chainer import FunctionSet, optimizers
from sklearn.base import ClassifierMixin
from sklearn import cross_validation
from sklearn.datasets import fetch_mldata

class MnistComplex(NNmanager, ClassifierMixin):
    def __init__(self, nnpacker, inspect=False):
        self.nnpacker = nnpacker
        # nnpackerの全ネットワーク構造を取得
        model = FunctionSet(**nnpacker.getFunctions())
        # 学習プロセスの設定
        optimizer = optimizers.SGD()
        lossFunction = F.softmax_cross_entropy
        params = {'epoch': 20, 'batchsize': 100, 'inspect': inspect}
        NNmanager.__init__(self, model, optimizer, lossFunction, **params)

    def trimOutput(self, output):
        y_trimed = output.data.argmax(axis=1)
        return np.array(y_trimed, dtype=np.int32)

    def forward(self, x_batch, **options):
        x_data = {'upper': x_batch[:, 0:392], 'lower': x_batch[:, 392:784]}
        return self.nnpacker.execute(x_data)
```

`forward`メソッドの中では、`NNpacker`インスタンスの`execute`メソッドを呼び出すようにします。
先ほどと同じように、Scikit-learnのモジュールを使って評価・検定を行うことができます。

```python
mnist = fetch_mldata('MNIST original')
x_all = mnist['data'].astype(np.float32) / 255
y_all = mnist['target'].astype(np.int32)

union = Union()
mc = MnistComplex(union, inspect=True)
score = cross_validation.cross_val_score(mc, x_all, y_all, cv=2)
print score.mean()
```

###Try Example
上記のサンプルコードは、`./examples/mnist_complex.py`で試すことができます

```shell
$ python ./examples/mnist_complex.py
```


#Test
テストは以下のコマンドで実行できます

```shell
$ pwd 
# /path/to/xchainer
$ python -m unittest discover -s tests
```

このテストでは、各機能についての動作検証を主な目的としているため、学習の反復数(epoch)が`5`と非常に短い設定になっています。実際に利用する際には、少なくとも20epoch以上の学習を行います。
  
```
# example output of test
Loading MNIST data for test. This could take a while...
...done

===Test `fit` method===
This could take a while...
...done

.===Test `predict` method. ===
This could take a while...
...done

.===Test learning with `cross_val_score` of sklearn.===
inspect learning process below...
[0 epoch] mean loss: 2.288577, mean accuracy: 0.116766
[1 epoch] mean loss: 2.241762, mean accuracy: 0.228158
[2 epoch] mean loss: 2.191640, mean accuracy: 0.331545
[3 epoch] mean loss: 2.132093, mean accuracy: 0.381466
[4 epoch] mean loss: 2.060875, mean accuracy: 0.422705
[0 epoch] mean loss: 2.338304, mean accuracy: 0.182222
[1 epoch] mean loss: 2.286201, mean accuracy: 0.244950
[2 epoch] mean loss: 2.225919, mean accuracy: 0.305917
[3 epoch] mean loss: 2.160249, mean accuracy: 0.333771
[4 epoch] mean loss: 2.090390, mean accuracy: 0.375406
[ 0.43661972  0.36954596]
...complete

.===Test `execute` method===
...done

.===Test `getFunctions` method===
...done

.===Test `insource` method===
...done

.===Test `outsource` method===
...done

.
----------------------------------------------------------------------
Ran 7 tests in 7.891s

OK

```

###Error in Loading MNIST data
このテストではScikit-learnのMNISTデータを利用していますが、お使いのマシンに古いMNISTデータがキャッシュされていると、データの読み込み時にエラーが発生する可能性があります。その際には、古いデータをマシンから削除してもう一度お試しください。

```shell
rm ~/scikit_learn_data/mldata/mnist-original.mat
```

##TODO
* README随時更新
