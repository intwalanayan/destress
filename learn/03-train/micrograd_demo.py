"""
Micrograd — a scalar autograd engine, from scratch, ~100 lines.

A compact rewrite of Andrej Karpathy's micrograd
(https://github.com/karpathy/micrograd) so you can read and run every
line yourself. This is the entire "backward pass" concept from Module 3,
made real — no framework magic.

What you'll see when you run this:
  1. We build a tiny expression: x1*w1 + x2*w2 + b, then apply an
     activation function (tanh — a "bend the number" step from Module 0).
  2. One forward pass produces an output value o.
  3. Calling o.backward() walks the expression graph backward and
     computes, for every input, "how much did I contribute to o?"
     — that's the gradient (Module 3 Concepts, section 4).
  4. Every leaf value now has a .grad attached. Positive means "nudging
     me up would raise o"; negative means "nudging me up would lower o".

Run:
    python learn/03-train/micrograd_demo.py
"""
from __future__ import annotations

import math


class Value:
    """One scalar number that remembers how it was computed."""

    def __init__(self, data, _children=(), _op=""):
        self.data = data                # the actual number
        self.grad = 0.0                 # gradient wrt to whatever we call .backward() on
        self._backward = lambda: None   # local backward — set by ops below
        self._prev = set(_children)     # inputs to this Value
        self._op = _op                  # for debugging

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:+.4f})"

    # -------- operations (each one defines its own local backward) --------

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1  → pass grad straight through
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a  → route grad × the other input
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float))
        out = Value(self.data ** exponent, (self,), f"**{exponent}")
        def _backward():
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        """Squash the input into (-1, 1). Common activation function."""
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward():
            # d(tanh(x))/dx = 1 - tanh(x)^2
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    # convenience so we can write -x, x-y, y*2, 5+x, etc.
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return (-self) + other
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * (other ** -1 if isinstance(other, Value) else Value(other) ** -1)

    # -------- the magic: topological order + walk backward --------
    def backward(self):
        # Build a topological order so we visit each node AFTER everything
        # downstream of it. This is why backward walks in reverse.
        topo = []
        seen = set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)

        # Seed: derivative of the output with respect to itself is 1.
        self.grad = 1.0

        # Walk backward — each node distributes its grad to its inputs.
        for v in reversed(topo):
            v._backward()


# ==============================================================
# Demo — one tiny "neuron" with tanh activation
# ==============================================================

def demo() -> None:
    print("=" * 60)
    print("Micrograd demo: a tiny 2-input 'neuron' with tanh activation")
    print("=" * 60)

    # Inputs (think: hours studied, hours slept from the Module 0 example)
    x1 = Value(2.0)
    x2 = Value(0.0)

    # Weights + bias (some random-looking values to start)
    w1 = Value(-3.0)
    w2 = Value(1.0)
    b  = Value(6.8813735870195432)   # this specific value makes o=0.7 exactly (Karpathy's trick)

    # Forward pass — the "chain of simple equations" from Module 0
    n = x1 * w1 + x2 * w2 + b
    o = n.tanh()

    print(f"\nForward pass produced output o = {o.data:.4f}")
    print("(This is the model's prediction with these current weights.)\n")

    # Backward pass — the whole point
    o.backward()

    print("Gradients (∂o/∂value = 'how much o changes per unit change in this value'):\n")
    for name, v in [("x1", x1), ("x2", x2), ("w1", w1), ("w2", w2), ("b", b)]:
        arrow = "  ↑ nudge UP raises o" if v.grad > 0 else ("  ↓ nudge UP lowers o" if v.grad < 0 else "  · no effect")
        print(f"  {name}.data = {v.data:>+7.3f}    grad = {v.grad:>+7.4f}{arrow}")

    print("\nRead this like a training-loop instruction:")
    print("  If we wanted o to be BIGGER, move weights in the direction of their grad.")
    print("  If we wanted o to be SMALLER, move weights AGAINST their grad.")
    print("  x2.grad is 0 because x2 itself is 0 — it had no influence on the output.")

    print("\nThat's step 3 of the training loop (backward pass) in ~100 lines of Python.")


if __name__ == "__main__":
    demo()
