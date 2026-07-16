"""
Side-by-side comparison: same prompt through the base Qwen3-8B and through
your fine-tuned pirate version.

This makes the impact of LoRA fine-tuning concrete. Same weights on disk,
same prompt, same sampler — the only difference is loading the tiny
adapter file on top.

Prereqs:
  - LoRA training done (adapter directory exists)

Run:
    python learn/04-finetune/compare_base_vs_lora.py
"""
from __future__ import annotations

from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

BASE_MODEL = "mlx-community/Qwen3-8B-4bit"
ADAPTER_PATH = Path(__file__).parent / "adapters"

PROMPTS = [
    "Hello!",
    "What's the capital of France?",
    "Tell me a joke.",
    "Explain gravity in one sentence.",
]


def reply(model, tokenizer, prompt: str, sampler) -> str:
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    return generate(model, tokenizer, prompt=prompt_text, max_tokens=150,
                    sampler=sampler, verbose=False).strip()


def main() -> None:
    if not ADAPTER_PATH.exists():
        print(f"No adapter at {ADAPTER_PATH}. Train LoRA first (Hands-on Step 2).")
        raise SystemExit(1)

    sampler = make_sampler(temp=0.7)

    print("Loading BASE model (no adapter)…")
    base_model, base_tokenizer = load(BASE_MODEL)

    print("Loading FINE-TUNED model (base + pirate adapter)…\n")
    ft_model, ft_tokenizer = load(BASE_MODEL, adapter_path=str(ADAPTER_PATH))

    for i, prompt in enumerate(PROMPTS, start=1):
        print(f"{'=' * 70}")
        print(f"Prompt {i}: {prompt}")
        print('=' * 70)

        base_reply = reply(base_model, base_tokenizer, prompt, sampler)
        ft_reply = reply(ft_model, ft_tokenizer, prompt, sampler)

        print(f"\n[BASE — no fine-tuning]")
        print(base_reply)

        print(f"\n[FINE-TUNED — pirate LoRA]")
        print(ft_reply)
        print()


if __name__ == "__main__":
    main()
