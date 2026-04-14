from __future__ import annotations

import os
import gzip
import random
import struct
import urllib.request
from typing import Tuple

import numpy as np


# ============================================================
# MNIST download / load
# ============================================================

MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def download_file(url: str, save_path: str) -> None:
    if os.path.exists(save_path):
        print(f"[skip] {save_path} already exists")
        return
    print(f"[download] {url}")
    urllib.request.urlretrieve(url, save_path)
    print(f"[saved] {save_path}")


def ensure_mnist(data_dir: str = "mnist_data") -> None:
    os.makedirs(data_dir, exist_ok=True)
    for _, url in MNIST_URLS.items():
        filename = os.path.join(data_dir, os.path.basename(url))
        download_file(url, filename)


def load_mnist_images(path: str) -> np.ndarray:
    """
    Return:
        images: shape (num_images, 784), dtype float32, normalized to [0, 1]
    """
    with gzip.open(path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid image file magic number: {magic}")

        image_size = rows * cols
        raw = f.read(num_images * image_size)

    images = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
    images = images.reshape(num_images, image_size) / 255.0
    return images


def load_mnist_labels(path: str) -> np.ndarray:
    """
    Return:
        labels: shape (num_items,), dtype int64
    """
    with gzip.open(path, "rb") as f:
        magic, num_items = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid label file magic number: {magic}")

        raw = f.read(num_items)

    labels = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
    return labels


# ============================================================
# Utility functions
# ============================================================

def one_hot(label: int, num_classes: int = 10) -> np.ndarray:
    v = np.zeros(num_classes, dtype=np.float32)
    v[label] = 1.0
    return v


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)

# objective function L
# epsilon is introduced to prevent the argument of the log from becoming zero
def cross_entropy(pred: np.ndarray, target: np.ndarray) -> float:
    eps = 1e-12
    return float(-np.sum(target * np.log(pred + eps)))


# ============================================================
# Neural network (2 hidden layers)
# ============================================================

class SimpleMLP2Hidden:
    """
    A small MLP for MNIST with 2 hidden layers:
        input -> hidden1 -> hidden2 -> output
        784   -> 128     -> 64      -> 10
    """

    def __init__(
        self,
        input_size: int,
        hidden_size1: int,
        hidden_size2: int,
        output_size: int,
        seed: int = 42,
    ) -> None:
        np.random.seed(seed)
        random.seed(seed)

        # Xavier-like initialization
        self.W1 = (
            (np.random.rand(input_size, hidden_size1).astype(np.float32) * 2.0 - 1.0)
            * np.sqrt(1.0 / input_size)
        )
        self.b1 = np.zeros(hidden_size1, dtype=np.float32)

        self.W2 = (
            (np.random.rand(hidden_size1, hidden_size2).astype(np.float32) * 2.0 - 1.0)
            * np.sqrt(1.0 / hidden_size1)
        )
        self.b2 = np.zeros(hidden_size2, dtype=np.float32)

        self.W3 = (
            (np.random.rand(hidden_size2, output_size).astype(np.float32) * 2.0 - 1.0)
            * np.sqrt(1.0 / hidden_size2)
        )
        self.b3 = np.zeros(output_size, dtype=np.float32)

    def forward(
        self,
        x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        x  : shape (784,)

        Returns:
            z1 : shape (128,)
            a1 : shape (128,)
            z2 : shape (64,)
            a2 : shape (64,)
            z3 : shape (10,)
            y  : shape (10,)
        """

        # hidden layer 1
        z1 = x @ self.W1 + self.b1
        a1 = sigmoid(z1)

        # hidden layer 2
        z2 = a1 @ self.W2 + self.b2
        a2 = sigmoid(z2)

        # output layer
        z3 = a2 @ self.W3 + self.b3
        y = softmax(z3)

        return z1, a1, z2, a2, z3, y

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        _, _, _, _, _, y = self.forward(x)
        return y

    def predict(self, x: np.ndarray) -> int:
        y = self.predict_proba(x)
        return int(np.argmax(y))

    def train_one(self, x: np.ndarray, label: int, learning_rate: float) -> float:
        """
        Train on one sample using SGD.
        """
        target = one_hot(label, self.b3.shape[0])

        # ----------------------------
        # forward
        # ----------------------------
        z1, a1, z2, a2, z3, y = self.forward(x)

        # ----------------------------
        # loss
        # ----------------------------
        loss = cross_entropy(y, target)

        # ----------------------------
        # backward
        # ----------------------------

        # output layer
        # softmax + cross entropy:
        # dL/dz3 = y - target
        dz3 = y - target                        # shape (10,)
        dW3 = np.outer(a2, dz3)                 # shape (64, 10)
        db3 = dz3.copy()                        # shape (10,)

        # hidden layer 2
        da2 = self.W3 @ dz3                     # shape (64,)
        dz2 = da2 * a2 * (1.0 - a2)             # shape (64,)
        dW2 = np.outer(a1, dz2)                 # shape (128, 64)
        db2 = dz2.copy()                        # shape (64,)

        # hidden layer 1
        da1 = self.W2 @ dz2                     # shape (128,)
        dz1 = da1 * a1 * (1.0 - a1)             # shape (128,)
        dW1 = np.outer(x, dz1)                  # shape (784, 128)
        db1 = dz1.copy()                        # shape (128,)

        # ----------------------------
        # SGD update
        # ----------------------------
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

        return loss


# ============================================================
# Evaluation
# ============================================================

def accuracy(
    model: SimpleMLP2Hidden,
    images: np.ndarray,
    labels: np.ndarray,
    max_samples: int | None = None,
) -> float:
    if max_samples is None or max_samples > len(images):
        max_samples = len(images)

    correct = 0
    for i in range(max_samples):
        pred = model.predict(images[i])
        if pred == int(labels[i]):
            correct += 1

    return correct / max_samples


def evaluate_loss(
    model: SimpleMLP2Hidden,
    images: np.ndarray,
    labels: np.ndarray,
    max_samples: int | None = None,
) -> float:
    if max_samples is None or max_samples > len(images):
        max_samples = len(images)

    total_loss = 0.0
    for i in range(max_samples):
        y = model.predict_proba(images[i])
        target = one_hot(int(labels[i]), 10)
        total_loss += cross_entropy(y, target)

    return total_loss / max_samples


# ============================================================
# Training
# ============================================================

def train(
    model: SimpleMLP2Hidden,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    epochs: int = 5,
    learning_rate: float = 0.05,
    train_subset: int = 3000,
    test_subset: int = 1000,
) -> None:
    n = min(train_subset, len(train_images))
    indices = np.arange(n)

    for epoch in range(1, epochs + 1):
        np.random.shuffle(indices)

        total_loss = 0.0

        for step, idx in enumerate(indices, start=1):
            x = train_images[idx]         # shape (784,)
            label = int(train_labels[idx])

            loss = model.train_one(x, label, learning_rate)
            total_loss += loss

            if step % 500 == 0:
                avg_loss = total_loss / step
                print(f"Epoch {epoch} Step {step}/{n} Loss {avg_loss:.4f}")

        avg_loss = total_loss / n
        train_acc = accuracy(model, train_images, train_labels, max_samples=min(1000, n))
        test_acc = accuracy(model, test_images, test_labels, max_samples=test_subset)
        test_loss = evaluate_loss(model, test_images, test_labels, max_samples=test_subset)

        print("=" * 60)
        print(f"Epoch {epoch} finished")
        print(f"Average train loss: {avg_loss:.4f}")
        print(f"Test loss         : {test_loss:.4f}")
        print(f"Train accuracy    : {train_acc * 100:.2f}%")
        print(f"Test accuracy     : {test_acc * 100:.2f}%")
        print("=" * 60)


# ============================================================
# Main
# ============================================================

def main() -> None:
    data_dir = "mnist_data"
    ensure_mnist(data_dir)

    train_images_path = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
    train_labels_path = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    test_images_path = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
    test_labels_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")

    print("[load] train images")
    train_images = load_mnist_images(train_images_path)

    print("[load] train labels")
    train_labels = load_mnist_labels(train_labels_path)

    print("[load] test images")
    test_images = load_mnist_images(test_images_path)

    print("[load] test labels")
    test_labels = load_mnist_labels(test_labels_path)

    print(f"train images shape: {train_images.shape}")
    print(f"train labels shape: {train_labels.shape}")
    print(f"test images shape : {test_images.shape}")
    print(f"test labels shape : {test_labels.shape}")

    model = SimpleMLP2Hidden(
        input_size=784,
        hidden_size1=128,
        hidden_size2=64,
        output_size=10,
        seed=42,
    )

    train(
        model,
        train_images,
        train_labels,
        test_images,
        test_labels,
        epochs=5,
        learning_rate=0.05,
        train_subset=3000,
        test_subset=1000,
    )

    print("\nSample predictions:")
    for i in range(10):
        pred = model.predict(test_images[i])
        true = int(test_labels[i])
        print(f"index={i:2d}  pred={pred}  true={true}")


if __name__ == "__main__":
    main()