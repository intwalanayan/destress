"""
Chat with the fine-tuned Qwen3-8B (base model + your pirate LoRA adapter).

Loads the base Qwen3-8B-4bit and points MLX-LM at the adapter directory
saved by `mlx_lm.lora --train`. The base model is unchanged on disk;
the adapter is the tiny file that adds the pirate style on top.

Prereqs:
  - LoRA training already run (see Module 4 Hands-on Step 2)
  - Adapter files exist under learn/04-finetune/adapters/

Run:
    python learn/04-finetune/chat_pirate.py
"""
from __future__ import annotations

from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

BASE_MODEL = "mlx-community/Qwen3-8B-4bit"
ADAPTER_PATH = Path(__file__).parent / "adapters"

# Try a few prompts to see the style transfer
PROMPTS = [
    "Hello, who are you?",
    "What's a good book to read?",
    "Can you explain machine learning to me?",
    "I'm feeling a bit down today.",
    "What's the weather like on Mars?",
]


def main() -> None:
    if not ADAPTER_PATH.exists():
        print(f"No adapter at {ADAPTER_PATH}. Run the LoRA training first (Hands-on Step 2).")
        raise SystemExit(1)

    print(f"Loading base:    {BASE_MODEL}")
    print(f"Loading adapter: {ADAPTER_PATH}")
    model, tokenizer = load(BASE_MODEL, adapter_path=str(ADAPTER_PATH))
    print("Loaded.\n")

    sampler = make_sampler(temp=0.7)

    for i, prompt in enumerate(PROMPTS, start=1):
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        response = generate(
            model, tokenizer,
            prompt=prompt_text,
            max_tokens=180,
            sampler=sampler,
            verbose=False,
        )

        print(f"=== {i}. USER ===")
        print(prompt)
        print(f"\n=== ASSISTANT (pirate-tuned) ===")
        print(response.strip())
        print()


if __name__ == "__main__":
    main()
