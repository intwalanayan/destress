"""
Peek at the raw logits — see the top-10 candidate next tokens BEFORE the sampler picks.

This runs exactly ONE forward pass through the model for each prompt, grabs the
logits (raw scores) at the last position, converts them to probabilities via
softmax, and prints the top 10 candidates with their probabilities and a
tiny bar chart.

Prereqs:
  - Model already downloaded (Module 1): mlx-community/Qwen3-8B-4bit

Run:
    python learn/02-model-internals/peek_logits.py
"""
from __future__ import annotations

import mlx.core as mx

from mlx_lm import load

MODEL = "mlx-community/Qwen3-8B-4bit"
TOP_K = 10

PROMPTS = [
    "The capital of France is",           # constrained — should have one dominant answer
    "2 + 2 =",                            # constrained — should be very sharp
    "Once upon a time,",                  # open-ended — many good candidates
    "The best pizza topping is",          # opinion — flat distribution
    "def hello():\n    ",                 # code — biased toward 'print'
]


def top_next_tokens(model, tokenizer, prompt: str, k: int = TOP_K) -> None:
    ids = tokenizer.encode(prompt)
    input_ids = mx.array(ids)[None]   # add batch dimension → shape [1, seq_len]

    # ONE forward pass through all 36 layers of the model.
    # Output shape: [1, seq_len, vocab_size] — a score for every vocab word at every position.
    logits = model(input_ids)

    # We only care about the LAST position (that's the one predicting the next token).
    last_logits = logits[0, -1, :]        # shape [vocab_size]

    # Softmax turns raw scores into probabilities that sum to 1.
    probs = mx.softmax(last_logits, axis=-1)

    # Get indices of the top-k probabilities.
    # mx.argsort is ascending, so we negate to get descending.
    top_indices = mx.argsort(-probs)[:k]

    # Force computation.
    mx.eval(probs, top_indices)

    idx_list = top_indices.tolist()

    print(f"\nPrompt:  {prompt!r}")
    print(f"Top {k} candidate next tokens (out of {tokenizer.vocab_size:,}):")
    for idx in idx_list:
        prob = float(probs[idx])
        piece = tokenizer.decode([idx])
        vis = piece.replace(" ", "·").replace("\n", "\\n")
        bar = "█" * int(prob * 50)
        print(f"  {vis:>18s}  {prob * 100:5.1f}%  {bar}")


def main() -> None:
    print(f"Loading {MODEL}…")
    model, tokenizer = load(MODEL)

    for prompt in PROMPTS:
        top_next_tokens(model, tokenizer, prompt)

    print()
    print("Notice:")
    print("  • Constrained prompts (facts, math) → one candidate with >50% probability.")
    print("  • Open-ended prompts → the top candidate might only be 5-15% — many plausible next words.")
    print("  • This is the raw distribution the sampler picks from. Temperature reshapes it.")


if __name__ == "__main__":
    main()
