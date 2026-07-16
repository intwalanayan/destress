"""
Train a small neural network on MNIST digits, on your Mac's GPU.

The FULL Module 3 training loop, made concrete:
  1. forward pass  → model(x) gives logits
  2. compute loss  → cross-entropy (classification task)
  3. backward pass → loss.backward() runs autograd
  4. optimizer step → Adam nudges every weight

At the end of each epoch, evaluates on the held-out test set so you
can see the validation accuracy climb (and spot overfitting if it kicks in).

Prereqs:
  - Module 0 environment complete
  - MPS available (verify_environment.py all PASS)
  - torchvision installed:  uv pip install torchvision

Run:
    python learn/03-train/mnist_train.py

Expected: ~1-2 minutes total on M3 Pro. Val accuracy climbs to ~97-98%.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DATA_DIR = Path.home() / ".cache" / "destress-mnist"
CKPT_PATH = Path(__file__).parent / "mnist_ckpt.pt"

BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3


def get_loaders() -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),   # standard MNIST mean/std
    ])
    train_ds = datasets.MNIST(str(DATA_DIR), train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(str(DATA_DIR), train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader


def build_model() -> nn.Module:
    """Simple MLP: 784 inputs (28×28 pixels flattened) → 10 outputs (one per digit class)."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128), nn.ReLU(),
        nn.Linear(128, 64),      nn.ReLU(),
        nn.Linear(64, 10),
    )


def eval_accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    model.train()
    return correct / total


def main() -> None:
    print(f"Device: {DEVICE}")
    if DEVICE == "cpu":
        print("WARNING: MPS not available; falling back to CPU (much slower).")

    print(f"Loading MNIST (first run downloads ~10 MB to {DATA_DIR})…")
    train_loader, test_loader = get_loaders()
    print(f"Training examples: {len(train_loader.dataset):,}")
    print(f"Test examples:     {len(test_loader.dataset):,}")

    model = build_model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: {BATCH_SIZE},  Epochs: {EPOCHS},  Learning rate: {LEARNING_RATE}\n")

    print(f"{'epoch':>5}  {'train loss':>10}  {'val acc':>8}  {'time (s)':>8}")
    print("-" * 42)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            # --- the 4-step training loop from Module 3 ---
            logits = model(x)                    # 1. forward pass
            loss = F.cross_entropy(logits, y)    # 2. compute loss
            optimizer.zero_grad()                #    (clear old grads first)
            loss.backward()                      # 3. backward pass (autograd)
            optimizer.step()                     # 4. optimizer nudges weights
            # -----------------------------------------------

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = eval_accuracy(model, test_loader) * 100
        elapsed = time.perf_counter() - t0
        print(f"{epoch:>5}  {avg_loss:>10.4f}  {val_acc:>7.2f}%  {elapsed:>8.1f}")

    torch.save(model.state_dict(), CKPT_PATH)
    print(f"\nCheckpoint saved to {CKPT_PATH}")
    print("Try: python learn/03-train/mnist_infer.py")


if __name__ == "__main__":
    main()
