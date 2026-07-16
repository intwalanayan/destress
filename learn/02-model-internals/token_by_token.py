"""
Watch an LLM generate a reply ONE TOKEN AT A TIME.

Instead of streaming the finished text, we consume `stream_generate` and print
each token as it emerges — so you can *see* the autoregressive loop from Module 2:
input → forward pass → one new token → append → repeat.

Prereqs:
  - Model already downloaded (Module 1): mlx-community/Qwen3-8B-4bit

Run:
    python learn/02-model-internals/token_by_token.py
"""
from __future__ import annotations

import time

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

MODEL = "mlx-community/Qwen3-8B-4bit"
PROMPT = "The capital of France is"
MAX_TOKENS = 40


def main() -> None:
    print(f"Loading {MODEL}…")
    model, tokenizer = load(MODEL)

    print(f"\nPrompt: {PROMPT!r}")
    prompt_ids = tokenizer.encode(PROMPT)
    print(f"Prompt tokens: {len(prompt_ids)}\n")

    print("Generating — one row per forward pass, showing what the model just produced:\n")
    print(f"  {'step':>4}  {'token id':>10}  {'piece':<20}  {'ms':>6}")
    print(f"  {'----':>4}  {'--------':>10}  {'-----':<20}  {'--':>6}")

    accumulated_text = ""
    last_time = time.perf_counter()

    stream = stream_generate(
        model,
        tokenizer,
        prompt=PROMPT,
        max_tokens=MAX_TOKENS,
        sampler=make_sampler(temp=0.0),   # greedy so it's reproducible
    )

    for step, chunk in enumerate(stream, start=1):
        now = time.perf_counter()
        dt_ms = (now - last_time) * 1000
        last_time = now

        piece = chunk.text
        # First result includes prompt-processing time; skip its huge "ms" for readability
        display_piece = piece.replace("\n", "\\n").replace(" ", "·")
        print(f"  {step:>4}  {chunk.token:>10d}  {display_piece:<20}  {dt_ms:>6.0f}")

        accumulated_text += piece

    print(f"\nFinal answer: {PROMPT}{accumulated_text}")
    print(f"\nEvery row above = one full forward pass through Qwen3-8B (all ~8B weights).")


if __name__ == "__main__":
    main()
