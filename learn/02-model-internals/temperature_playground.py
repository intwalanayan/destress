"""
Feel the temperature knob.

Same prompt, three temperatures, three runs each. See how outputs go from
identical-and-boring (temp=0) to varied-and-creative (temp=0.7) to
loose-and-often-weird (temp=1.5).

Prereqs:
  - Model already downloaded (Module 1): mlx-community/Qwen3-8B-4bit

Run:
    python learn/02-model-internals/temperature_playground.py
"""
from __future__ import annotations

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

MODEL = "mlx-community/Qwen3-8B-4bit"
PROMPT = "Describe the color blue in one sentence."
MAX_TOKENS = 50


def try_temperature(model, tokenizer, temp: float, runs: int = 3) -> None:
    print(f"\n{'=' * 70}")
    print(f"temperature = {temp}")
    print(f"{'=' * 70}")

    for run in range(1, runs + 1):
        response = generate(
            model, tokenizer,
            prompt=PROMPT,
            max_tokens=MAX_TOKENS,
            sampler=make_sampler(temp=temp),
            verbose=False,
        )
        # Clean up whitespace for one-line display
        response = response.strip().replace("\n", " ")
        print(f"  run {run}:  {response}")


def main() -> None:
    print(f"Loading {MODEL}…")
    model, tokenizer = load(MODEL)

    print(f"\nPrompt (same every time): {PROMPT!r}")

    for temp in (0.0, 0.7, 1.5):
        try_temperature(model, tokenizer, temp)

    print()
    print("Notice:")
    print("  temp = 0.0 → identical every run (greedy — always picks the top token)")
    print("  temp = 0.7 → varied but sensible (typical chat default)")
    print("  temp = 1.5 → distribution is flattened; low-probability tokens creep in")


if __name__ == "__main__":
    main()
