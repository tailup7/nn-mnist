# MNIST Digit Classification with Neural Networks
本リポジトリは、手書き数字データセット（MNIST）を用いた多層パーセプトロン（MLP）による多クラス分類モデルの実装です。

[Chainer Tutrial](https://tutorials.chainer.org/ja/13_Basics_of_Neural_Networks.html) を参考にしています。

## 問題設定

入力画像をベクトルとして扱い、
次のような関数を学習する：

$$
f: \mathbb{R}^{784} \to {0,1,2,\dots,9}
$$

ここで、

* 入力画像のpixelサイズ： $M = 28 \times 28 = 784$ pixel
* 出力：数字クラス（0〜9）($K=10$)
* サンプルサイズ (入力画像のデータ数) : $N$

各pixel ($\mathbf{x}$の各成分) は0~255の整数値を取る (0:黒、255: 白)。ただしコード内では正規化して扱う。

## モデル（関数の構造）

本コードでは、次のような関数合成を行っている：

$$
\mathbf{x} \in \mathbb{R}^{784}
$$

$$
\mathbf{u}_1 = \sigma(\mathrm{W}_1 \mathbf{x} + \mathbf{b}_1)
$$

$$
\mathbf{u}_2 = \sigma(\mathrm{W}_2 \mathbf{u}_1 + \mathbf{b}_2)
$$

$$
\mathbf{y} = \mathrm{W}_3 \mathbf{u}_2 + \mathbf{b}_3
$$

$$
\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{y})
$$

中間層は2層で、 784 → 128 → 64 → 10

線形変換の結果として、1層目の中間層で得られる出力値 $\mathbf{u}_1$ は

$$
\mathbf{u}_1 = \mathrm{W}_1 \mathbf{x} + \mathbf{b}_1
$$

ここで、入力層 → 中間層1 の線形変換における重み行列 $\mathrm{W}_1$ のサイズは、中間層1 のノードの数を $n_1$ とすると
$n_1 \times M$ である。(今回は $n_1=128$, $M=784$ ) また、
$\mathbf{b}_1 \in \mathbb{R}^{128}$ はバイアスベクトルである。

## 各層の意味

### 1. 線形変換

$$
W \mathbf{x} + \mathbf{b}
$$

これは重回帰モデル：

$$
y = X\beta
$$

の非線形拡張です。

---

### 2. 非線形変換（ReLU）

$$
\sigma(x) = \max(0, x)
$$

<!---
👉 線形モデルでは表現できない関数を扱うために必要
--->
---

### 3. 出力（確率化）

$$
\mathrm{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{10} e^{z_j}}
$$

<!---
👉 各クラスの確率に変換
--->


## 損失関数（目的関数）

分類問題なので、目的関数(objective function)として次の交差エントロピー(cross entropy) を用いる

$$
L = - \sum_{k=1}^{10} y_k \log \hat{y}_k
$$

ここで、 $y_k$ は正解ラベル、 $\hat{y}_k$ はクラス $k$ に対する予測確率を表す。

これを最小にする重み行列 (今回は $\mathrm{W}_1$と $\mathrm{W}_2$と $\mathrm{W}_3$の3つ) およびバイアスベクトル (今回は $\mathbf{b}_1$と $\mathbf{b}_2$と $\mathbf{b}_3$の3つ) を求める。

<!--
* クロスエントロピー
* ロジスティック回帰の多クラス版
-->


## 学習（最適化）

パラメータ $\theta = {W, b}$ を更新：

$$
\theta \leftarrow \theta - \eta \nabla_\theta L
$$

ここで：

* $\eta$：学習率
* $\nabla_\theta L$：勾配

目的関数 

---

### 実装対応

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

---

## 実行方法

```bash
pip install torch torchvision
python mnist_torch.py
```

---

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
