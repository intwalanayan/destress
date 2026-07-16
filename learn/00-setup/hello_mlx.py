"""
First contact with MLX — Apple's native ML framework.

MLX targets Apple Silicon and its unified memory (CPU and GPU share the
same RAM). Notice: no 'device' argument anywhere — MLX picks the Apple
GPU by default.

MLX is 'lazy': operations only run when you ask for a result via
mx.eval() or .item(). This is why we call mx.eval() before timing.

Run:
    python learn/00-setup/hello_mlx.py
"""
from __future__ import annotations

import time

import mlx.core as mx


def timed_matmul(size: int = 2048, repeats: int = 10) -> float:
    a = mx.random.normal((size, size))
    b = mx.random.normal((size, size))
    # Warm up: force the first op to actually execute.
    mx.eval(a @ b)
    t0 = time.perf_counter()
    for _ in range(repeats):
        c = a @ b
    mx.eval(c)  # force the batch of matmuls to finish before stopping the clock
    return time.perf_counter() - t0


def main() -> None:
    print("MLX runs on the Apple GPU by default — unified memory means no .to(device).\n")

    size, repeats = 2048, 10
    print(f"Multiplying {size}x{size} matrices, {repeats} times:")
    t = timed_matmul(size=size, repeats=repeats)
    print(f"  MLX:  {t * 1000:>7.0f} ms")

    print("\nNo 'device' argument anywhere in this file — that is the MLX advantage.")
    print("Module 4 (fine-tuning) will use MLX for exactly this reason.")


if __name__ == "__main__":
    main()
