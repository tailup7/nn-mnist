"""
MNIST classifier with PyTorch

Features:
- Automatically downloads MNIST
- Uses a simple MLP: 784 -> 128 -> 64 -> 10
- Trains and evaluates the model
- Saves the trained model
- Can predict some test samples after training

"""

from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# Configuration
# ============================================================

SEED = 42
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3
DATA_DIR = Path("mnist_data")
MODEL_PATH = Path("mnist_mlp.pth")


# ============================================================
# Utility
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Model
# ============================================================

class MNISTMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),          # (N, 1, 28, 28) -> (N, 784)
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Data
# ============================================================

def get_dataloaders(batch_size: int = 64) -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    # ToTensor():
    #   PIL image / ndarray [0,255] -> float tensor [0.0, 1.0]

    train_dataset = datasets.MNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, test_loader


# ============================================================
# Train / Evaluate
# ============================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)              # shape: (batch_size, 10)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ============================================================
# Save / Load
# ============================================================

def save_model(model: nn.Module, path: Path) -> None:
    torch.save(model.state_dict(), path)
    print(f"[saved] model -> {path}")


def load_model(path: Path, device: torch.device) -> MNISTMLP:
    model = MNISTMLP().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


# ============================================================
# Prediction demo
# ============================================================

@torch.no_grad()
def show_sample_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = 10,
) -> None:
    model.eval()

    shown = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        for i in range(images.size(0)):
            print(
                f"sample {shown:2d}: "
                f"pred = {preds[i].item()} / true = {labels[i].item()}"
            )
            shown += 1
            if shown >= num_samples:
                return


# ============================================================
# Main
# ============================================================

def main() -> None:
    set_seed(SEED)
    device = get_device()
    print(f"device: {device}")

    train_loader, test_loader = get_dataloaders(BATCH_SIZE)

    model = MNISTMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(model)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        test_loss, test_acc = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
        )

        print("-" * 60)
        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
        )
        print("-" * 60)

    save_model(model, MODEL_PATH)

    print("\nSample predictions on test data:")
    show_sample_predictions(model, test_loader, device, num_samples=10)


if __name__ == "__main__":
    main()