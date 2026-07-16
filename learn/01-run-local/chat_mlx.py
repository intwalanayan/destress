"""
Chat with a local model via MLX-LM directly — no HTTP layer, no server.

Uses MLX's Apple-Silicon-native inference path, the fastest option on Macs
below 32 GB (which is where Ollama's own MLX backend kicks in).

Prerequisites:
  - mlx-lm installed (Module 0):  `uv pip install mlx-lm`
  - First run downloads the model (~5 GB) into ~/.cache/huggingface/

Run:
    python learn/01-run-local/chat_mlx.py
"""
from __future__ import annotations

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

MODEL = "mlx-community/Qwen3-8B-4bit"

PROMPT = "Explain LoRA in 2 sentences, aimed at a software engineer new to ML."


def main() -> None:
    print(f"Loading {MODEL}")
    print("(first run downloads ~5 GB; subsequent runs load from disk cache)\n")

    model, tokenizer = load(MODEL)

    # Apply the model's chat template so it knows the message is from a user.
    messages = [
        {"role": "system", "content": "You are a concise, friendly assistant."},
        {"role": "user", "content": PROMPT},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    print("Generating…\n")
    # verbose=True prints tokens as they're produced and shows tok/s at the end.
    generate(
        model,
        tokenizer,
        prompt=prompt_text,
        max_tokens=300,
        sampler=make_sampler(temp=0.7),
        verbose=True,
    )


if __name__ == "__main__":
    main()
