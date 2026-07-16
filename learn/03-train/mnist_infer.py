"""
Use the trained MNIST model on individual test digits.

Loads the checkpoint from mnist_train.py and classifies the first 10 test-set
images. Prints the model's guess vs the true label plus the confidence.

Prereqs:
  - mnist_train.py has run and saved mnist_ckpt.pt

Run:
    python learn/03-train/mnist_infer.py
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DATA_DIR = Path.home() / ".cache" / "destress-mnist"
CKPT_PATH = Path(__file__).parent / "mnist_ckpt.pt"


def build_model() -> nn.Module:
    """Must match the architecture in mnist_train.py."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128), nn.ReLU(),
        nn.Linear(128, 64),      nn.ReLU(),
        nn.Linear(64, 10),
    )


def main() -> None:
    if not CKPT_PATH.exists():
        print(f"No checkpoint at {CKPT_PATH}. Run mnist_train.py first.")
        raise SystemExit(1)

    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_ds = datasets.MNIST(str(DATA_DIR), train=False, download=True, transform=transform)

    print(f"Testing on the first 10 digits in the test set…\n")
    print(f"  {'idx':>3}  {'true':>4}  {'predicted':>9}  {'confidence':>10}  result")
    print("-" * 46)

    correct = 0
    with torch.no_grad():
        for i in range(10):
            x, true_label = test_ds[i]
            x = x.to(DEVICE).unsqueeze(0)          # add batch dim → [1, 1, 28, 28]
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze()
            pred = int(probs.argmax().item())
            prob = float(probs[pred])
            mark = "ok" if pred == true_label else "miss"
            if pred == true_label:
                correct += 1
            print(f"  {i:>3}  {true_label:>4}  {pred:>9}  {prob*100:>9.1f}%  {mark}")

    print(f"\n{correct}/10 correct on this small sample.")


if __name__ == "__main__":
    main()
