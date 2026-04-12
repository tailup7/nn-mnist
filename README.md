# MNIST Neural Network Practice

<!---

# MNIST Digit Classification with Neural Networks

## 概要

本リポジトリでは、手書き数字データセット（MNIST）を用いて、多層パーセプトロン（MLP）による分類モデルを実装しています。


---

## 問題設定

入力画像をベクトルとして扱い、
次のような関数を学習します：

$$
f: \mathbb{R}^{784} \to {0,1,2,\dots,9}
$$

ここで、

* 画像サイズ： $28 \times 28 = 784$ pixel
* 出力：数字クラス（0〜9）

---

## モデル（関数の構造）

本コードでは、次のような関数合成を行っています：

$$
\mathbf{x} \in \mathbb{R}^{784}
$$

$$
\mathbf{h}_1 = \sigma(W_1 \mathbf{x} + \mathbf{b}_1)
$$

$$
\mathbf{h}_2 = \sigma(W_2 \mathbf{h}_1 + \mathbf{b}_2)
$$

$$
\mathbf{z} = W_3 \mathbf{h}_2 + \mathbf{b}_3
$$

$$
\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{z})
$$

---

## 各層の意味

### 1. 線形変換（重回帰と同じ）

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

👉 線形モデルでは表現できない関数を扱うために必要

---

### 3. 出力（確率化）

$$
\mathrm{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{10} e^{z_j}}
$$

👉 各クラスの確率に変換

---

## 損失関数（目的関数）

分類問題なので、次の関数を最小化します：

$$
L = - \sum_{i=1}^{10} y_i \log \hat{y}_i
$$

これは：

* クロスエントロピー
* ロジスティック回帰の多クラス版

---

## 学習（最適化）

パラメータ $\theta = {W, b}$ を更新：

$$
\theta \leftarrow \theta - \eta \nabla_\theta L
$$

ここで：

* $\eta$：学習率
* $\nabla_\theta L$：勾配

---

### 実装対応

コードでは以下に対応：

* 勾配計算：自動微分
* 更新：Adam法

---

## データ処理

画像は次のように変換されます：

$$
[0,255] \rightarrow [0,1]
$$

さらに：

$$
28 \times 28 \rightarrow 784
$$

---

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
