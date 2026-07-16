"""
Phase 1 environment check.

Confirms:
  - macOS on Apple Silicon
  - Python 3.10+
  - Core ML packages installed (numpy, torch, mlx)
  - PyTorch can talk to the Apple GPU via MPS
  - MLX can run a matmul on the Apple GPU

Run:
    python learn/00-setup/verify_environment.py
"""
from __future__ import annotations

import platform
import sys


PASS = "[PASS]"
FAIL = "[FAIL]"


def check(label: str, ok: bool, detail: str = "") -> bool:
    tag = PASS if ok else FAIL
    line = f"{tag} {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    return ok


def section(title: str) -> None:
    print(f"\n== {title} ==")


def main() -> int:
    print("Phase 1 environment check")
    print("=" * 60)

    section("System")
    check("macOS", platform.system() == "Darwin", platform.mac_ver()[0] or "unknown")
    check("Apple Silicon (arm64)", platform.machine() == "arm64", platform.machine())
    check(
        "Python 3.10+",
        sys.version_info >= (3, 10),
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )

    section("Core packages")

    numpy_ok = False
    try:
        import numpy as np
        numpy_ok = check("numpy", True, np.__version__)
    except ImportError:
        check("numpy", False, "not installed — run: uv pip install numpy")

    torch_ok = False
    try:
        import torch
        torch_ok = check("torch (PyTorch)", True, torch.__version__)
    except ImportError:
        check("torch (PyTorch)", False, "not installed — run: uv pip install torch")

    mlx_ok = False
    try:
        import mlx.core as mx
        version = getattr(mx, "__version__", "installed")
        mlx_ok = check("mlx (Apple ML framework)", True, version)
    except ImportError:
        check("mlx (Apple ML framework)", False, "not installed — run: uv pip install mlx")

    section("Apple GPU access")

    if torch_ok:
        import torch
        mps_avail = torch.backends.mps.is_available()
        check("torch: MPS available", mps_avail)
        check("torch: MPS built", torch.backends.mps.is_built())
        if mps_avail:
            try:
                x = torch.randn(512, 512, device="mps")
                y = torch.randn(512, 512, device="mps")
                z = (x @ y).sum().item()
                check("torch: tiny matmul on MPS", True, f"result≈{z:.1f}")
            except Exception as e:
                check("torch: tiny matmul on MPS", False, str(e))
    else:
        print("(skipping torch GPU checks — torch not installed)")

    if mlx_ok:
        try:
            import mlx.core as mx
            x = mx.random.normal((512, 512))
            y = mx.random.normal((512, 512))
            z = (x @ y).sum().item()
            check("mlx: tiny matmul on Apple GPU", True, f"result≈{z:.1f}")
        except Exception as e:
            check("mlx: tiny matmul on Apple GPU", False, str(e))
    else:
        print("(skipping mlx GPU checks — mlx not installed)")

    print("\nDone. All PASS means your environment is ready for Module 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
