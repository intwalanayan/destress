"""
Head-to-head speed test: same prompt, same model family, through
  (a) Ollama's HTTP API — llama.cpp/Metal path on 18 GB Macs, and
  (b) MLX-LM directly  — Apple-native MLX path.

Prints tokens/second for each. On sub-32 GB Macs, MLX should win by 15-40%.

Prerequisites:
  - Ollama running with qwen3:8b pulled
  - mlx-lm installed
  - openai Python package installed
  - First run of the MLX side downloads ~5 GB.

Run:
    python learn/01-run-local/speed_compare.py
"""
from __future__ import annotations

import time

from openai import OpenAI

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

OLLAMA_MODEL = "qwen3:8b"
MLX_MODEL = "mlx-community/Qwen3-8B-4bit"

PROMPT = (
    "Write a ~200-word plain-English explainer of what quantization is in the "
    "context of neural networks. Aimed at a beginner. No formulas."
)
MAX_TOKENS = 300


def bench_ollama() -> tuple[int, float]:
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=MAX_TOKENS,
    )
    elapsed = time.perf_counter() - t0
    tokens = resp.usage.completion_tokens if resp.usage else MAX_TOKENS
    return tokens, elapsed


def bench_mlx() -> tuple[int, float]:
    print(f"Loading {MLX_MODEL} (cached after first run)…")
    model, tokenizer = load(MLX_MODEL)
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        tokenize=False,
    )
    t0 = time.perf_counter()
    response = generate(
        model,
        tokenizer,
        prompt=prompt_text,
        max_tokens=MAX_TOKENS,
        sampler=make_sampler(temp=0.7),
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    tokens = len(tokenizer.encode(response))
    return tokens, elapsed


def report(name: str, tokens: int, seconds: float) -> float:
    tps = tokens / seconds if seconds > 0 else 0.0
    print(f"  {name:<24s}  {tokens:>4d} tokens in {seconds:>5.1f}s  →  {tps:>5.1f} tok/s")
    return tps


def main() -> None:
    print("=" * 66)
    print("Speed comparison: Ollama (llama.cpp/Metal) vs MLX-LM (native)")
    print("=" * 66)
    print(f"Prompt:  {PROMPT[:80]}…")
    print(f"Model:   Qwen3-8B, 4-bit, on your M3 Pro 18 GB")
    print()

    print("Running Ollama…")
    ollama_tok, ollama_t = bench_ollama()
    print()
    print("Running MLX-LM…")
    mlx_tok, mlx_t = bench_mlx()

    print()
    print("Results:")
    ollama_tps = report("Ollama (llama.cpp)", ollama_tok, ollama_t)
    mlx_tps = report("MLX-LM (Apple-native)", mlx_tok, mlx_t)

    if ollama_tps > 0:
        ratio = mlx_tps / ollama_tps
        print()
        print(f"MLX-LM is {ratio:.2f}x the speed of Ollama on this workload.")
        print("(That's the 'MLX advantage' the Concepts section talked about.)")


if __name__ == "__main__":
    main()
