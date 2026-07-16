"""
First contact with the Apple GPU from PyTorch.

Multiplies two large matrices on the CPU and on the Apple GPU (MPS) and
prints how long each took. Purpose: prove the GPU works and *feel* the
speed difference so future 'device="mps"' code has real meaning to you.

Run:
    python learn/00-setup/hello_mps.py
"""
from __future__ import annotations

import time

import torch


def timed_matmul(device: str, size: int = 2048, repeats: int = 10) -> float:
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    # Warm up: first op on GPU includes one-time compilation.
    _ = (a @ b).sum().item()
    t0 = time.perf_counter()
    for _ in range(repeats):
        c = a @ b
    # .item() forces async GPU work to finish before we stop the clock.
    _ = c.sum().item()
    return time.perf_counter() - t0


def main() -> None:
    if not torch.backends.mps.is_available():
        print("Apple GPU (MPS) is not available. Run verify_environment.py first.")
        raise SystemExit(1)

    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available:   {torch.backends.mps.is_available()}\n")

    size, repeats = 2048, 10
    print(f"Multiplying {size}x{size} matrices, {repeats} times each:")

    cpu_time = timed_matmul("cpu", size=size, repeats=repeats)
    mps_time = timed_matmul("mps", size=size, repeats=repeats)

    print(f"  CPU:  {cpu_time * 1000:>7.0f} ms")
    print(f"  MPS:  {mps_time * 1000:>7.0f} ms")
    print(f"\nApple GPU is ~{cpu_time / mps_time:.0f}x faster than the CPU for this workload.")
    print('Whenever you later see device="mps" in code, this is what it is using.')


if __name__ == "__main__":
    main()
