"""
Poke a real tokenizer — see how text becomes token IDs.

Uses Qwen3-8B's official tokenizer (the same one our chat model uses).
Only tokenizer files download (~15 MB), not the model weights.

Prereqs:
  - uv pip install transformers
  - Ollama or MLX model from Module 1 (we reuse its tokenizer files)

Run:
    python learn/02-model-internals/tokenizer_playground.py
"""
from __future__ import annotations

from transformers import AutoTokenizer

TOKENIZER_SRC = "mlx-community/Qwen3-8B-4bit"


def show(tok, text: str) -> None:
    ids = tok.encode(text, add_special_tokens=False)
    pieces = [tok.decode([i]) for i in ids]

    print(f"\nInput  ({len(text)} chars):  {text!r}")
    print(f"Tokens ({len(ids):>3d}):")
    for tok_id, piece in zip(ids, pieces):
        # Show leading spaces visibly as '·' — otherwise they're invisible in terminal
        vis = piece.replace(" ", "·")
        print(f"  {tok_id:>7d}  →  '{vis}'")


def main() -> None:
    print(f"Loading tokenizer from {TOKENIZER_SRC}…")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_SRC)
    print(f"Vocabulary size: {tok.vocab_size:,} tokens")

    # Simple English
    show(tok, "Hello, world!")

    # A whole sentence — count tokens vs words
    show(tok, "The quick brown fox jumps over the lazy dog.")

    # Rare / made-up compounds get split
    show(tok, "I love running LLMs locally on my MacBook.")

    # Same word with & without leading space — different tokens
    show(tok, "cat")
    show(tok, " cat")

    # Emoji — often multi-byte, multi-token
    print("\n\n--- non-English / special content ---")
    show(tok, "🚀 EEG signals 📊")

    # Code
    show(tok, "def hello():\n    print('world')")

    # Hindi (Devanagari script)
    show(tok, "नमस्ते")

    # Japanese
    show(tok, "こんにちは")

    print("\n\nNotes to notice:")
    print("  • Common English words = 1 token. Rare or made-up words split into pieces.")
    print("  • Leading space is PART of the token — 'cat' and ' cat' are different IDs.")
    print("  • Emoji and non-Latin scripts often take MANY tokens per character.")
    print("  • Rule of thumb: 1 token ≈ ¾ of an English word. Different for other languages.")


if __name__ == "__main__":
    main()
