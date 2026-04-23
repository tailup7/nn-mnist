# MNIST Digit Classification with Neural Networks
本リポジトリは、手書き数字データセット（MNIST）を用いた多層パーセプトロン（MLP）による多クラス分類モデルの実装です。

[Chainer Tutrial](https://tutorials.chainer.org/ja/13_Basics_of_Neural_Networks.html) を参考にしています。

# 問題設定

入力画像をベクトルとして扱い、
次のようなモデルを学習する：

$$
f: \mathbb{R}^{784} \to {0,1,2,\dots,9}
$$

ここで、

* 入力画像のpixelサイズ： $M = 28 \times 28 = 784$ pixel
* 出力：数字クラス（0〜9）($K=10$)
* サンプルサイズ (入力画像のデータ数) : $N$

各pixel ($\mathbf{x}$の各成分) は0~255の整数値を取る (0:黒、255: 白)。ただしコード内では正規化して扱う。

# ニューラルネットワークによる推論

次のような関数を用いて**推論**を行う。

**推論** ... 訓練が完了したモデルを用いて新しいデータに対して予測を行うこと

$$
\mathbf{x} \in \mathbb{R}^{784}
$$

$$
\mathbf{u}_1 = \mathrm{W}_1 \mathbf{x} + \mathbf{b}_1
$$

$$
\mathbf{h}_1 = \sigma(\mathbf{u_1})
$$

$$
\mathbf{u}_2 = \mathrm{W}_2 \mathbf{h}_1 + \mathbf{b}_2
$$

$$
\mathbf{h}_2 = \sigma(\mathbf{u_2})
$$

$$
\mathbf{y} = \mathrm{W}_3 \mathbf{h}_2 + \mathbf{b}_3
$$

$$
\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{y})
$$

中間層は2層で、 784 → 128 → 64 → 10

重み行列 $\mathrm{W}_1$ のサイズは、中間層1 のノードの数を $n_1$ とすると
$n_1 \times M$ である (今回は $n_1=128$, $M=784$ )。 また、
$\mathbf{b}_1 \in \mathbb{R}^{n_1}$ はバイアスベクトルである。 同様に、中間層2のノード数を $n_2$とすると(今回は $n_2=64$)、 $\mathrm{W}_2$は $n_2 \times n_1$行列、 $\mathbf{b}_2 \in \mathbb{R}^{n_2}$ 、 $\mathrm{W}_3$は $K \times n_2$行列、 $\mathbf{b}_3 \in \mathbb{R}^{K}$ 

$\sigma$ はシグモイド関数であり、以下の式で表される。

$$
\sigma (x) = \dfrac{1}{1+e^{-x}}
$$

シグモイド関数は、ニューラルネットワークに非線形性を入れるための**活性化関数**の1つである。

また、 $\mathrm{softmax}$ 関数は、出力値を確率に変換するための正規化関数であり、ベクトル $\mathbf{y} = (y_1, y_2, ... , y_K)$ の各成分に対して以下の式で定義される。

$$
\hat{y_i} = \mathrm{softmax}(y_i) = \frac{e^{y_i}}{\sum_{k=1}^{K} e^{y_k}}
$$

# ニューラルネットワークの訓練

### 重み行列とバイアスベクトルの初期値

重み行列とバイアスベクトルの各成分に対して適当な初期値を設定する。今回は、重み行列の各成分は一様乱数を入力サイズでスケーリングした値、バイアスベクトルの各成分はすべて0で初期化する。

$$
\mathrm{W}_1 の各成分 \sim \mathrm{Uniform} (-1, 1) ×  \sqrt{1 / 784}
$$

$$
\mathrm{W}_2 の各成分 \sim \mathrm{Uniform} (-1, 1) ×  \sqrt{1 / 128}
$$

$$
\mathrm{W}_3 の各成分 \sim \mathrm{Uniform} (-1, 1) ×  \sqrt{1 / 64}
$$

### 損失関数（目的関数）

分類問題なので、目的関数(objective function)として次の交差エントロピー(cross entropy) を用いる

$$
L = - \sum_{k=1}^K t_k \log \hat{y}_k
$$

ここで、 $\mathbf{t}$ は正解ラベルであり、 $t_k (k = 1,2, ..., K)$ のいずれか1つだけが1で、それ以外が0であるようなベクトル(ワンホットベクトル)である。また、 $\hat{y}_k$ はクラス $k$ に対する予測確率を表す。 $L$の値が小さいほど、正解との誤差が小さい。

これを最小にする重み行列 (今回は $\mathrm{W}_1$と $\mathrm{W}_2$と $\mathrm{W}_3$の3つ) およびバイアスベクトル (今回は $\mathbf{b}_1$と $\mathbf{b}_2$と $\mathbf{b}_3$の3つ) を求める。

### パラメータの最適化

<!--
**誤差逆伝播法 (backpropagation)** によって勾配を求め、**確率的勾配降下法 (SGD: stocastic gradient descent)** (ミニバッチ学習を用いた勾配降下法) で更新する。
-->

$L(\mathrm{W}_1, \mathrm{W}_2, \mathrm{W}_3, \mathbf{b}_1, \mathbf{b}_2, \mathbf{b}_3)$ を最小にする $\mathrm{W}_1, \mathrm{W}_2, \mathrm{W}_3, \mathbf{b}_1, \mathbf{b}_2, \mathbf{b}_3$を決定する。 $\mathrm{W}_1, \mathrm{W}_2, \mathrm{W}_3, \mathbf{b}_1, \mathbf{b}_2, \mathbf{b}_3$はそれぞれ独立変数なので、それぞれの変数で $L$を偏微分すればよい。

まず、 $\partial L / \partial W_3$について、連鎖則を用いて

$$
\dfrac{\partial L}{\partial W_3} = \dfrac{\partial L}{\partial \mathbf{y}} \dfrac{\partial \mathbf{y}}{\partial W_3}
$$

<!--
パラメータ $\theta = {W, b}$ を更新：

$$
\theta \leftarrow \theta - \eta \nabla_\theta L
$$

ここで：

* $\eta$：学習率
* $\nabla_\theta L$：勾配
-->

### ミニバッチ学習とは

エポック ( `epochs` )

## 実装対応

コードでは以下に対応：

* 勾配計算：自動微分
* 更新：Adam法

## 精度の定義

予測：

$$
\hat{y} = \arg\max_i z_i
$$

精度：

$$
\text{Accuracy} = \frac{\text{正解数}}{\text{全データ数}}
$$

---

## コード構成

* モデル定義（MLP）
* 学習ループ
* 評価関数
* モデル保存
* 推論サンプル表示

<!--
## 実行方法

```bash
pip install torch torchvision
python mnist_torch.py
```
-->


<!---
## 数学的な解釈まとめ

このモデルは本質的には：

👉 **非線形基底関数を用いた重回帰モデル**

と考えることができます。

$$
f(\mathbf{x}) = W_3 \sigma(W_2 \sigma(W_1 \mathbf{x}))
$$

---

## 補足

このコードは以下のような特徴を持ちます：

* シンプルな全結合ネットワーク
* CNNは未使用
* 教育・入門用に最適

---

## 参考

* MNISTデータセット
* 深層学習入門

---
