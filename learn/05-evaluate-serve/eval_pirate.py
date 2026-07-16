"""
Evaluate the pirate fine-tune on held-out prompts (not seen during training).

For each test prompt, generates a response with:
  1. Base Qwen3-8B (no adapter)
  2. Fine-tuned pirate model (base + LoRA adapter)

Scores each response for "pirate-ness" using a small dictionary of pirate
marker words (arr, matey, aye, ye, ahoy, avast, savvy, ...).
Prints side-by-side comparison + aggregate score ratio.

Higher pirate-ness on the fine-tuned side = successful style transfer.

Prereqs:
  - LoRA training done (Module 4 Step 2); adapters at learn/04-finetune/adapters/

Run:
    python learn/05-evaluate-serve/eval_pirate.py
"""
from __future__ import annotations

import re
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


BASE_MODEL = "mlx-community/Qwen3-8B-4bit"
ADAPTER_PATH = Path(__file__).parent.parent / "04-finetune" / "adapters"
TEST_PROMPTS_FILE = Path(__file__).parent / "test_prompts.txt"

# Words that reliably signal pirate speech
PIRATE_MARKERS = [
    "arr", "arrr", "aye", "matey", "mateys", "ye", "yer",
    "ahoy", "avast", "tis", "twas", "scallywag", "landlubber",
    "hearty", "savvy", "pirate", "pirates", "plank", "shanty",
    "treasure", "seafarin", "seafaring", "grog", "shipmate",
]


def score_pirate_ness(text: str) -> int:
    """Count how many pirate marker words appear (word-boundary matching)."""
    text_lower = text.lower()
    count = 0
    for marker in PIRATE_MARKERS:
        pattern = r"\b" + re.escape(marker) + r"\b"
        count += len(re.findall(pattern, text_lower))
    return count


def reply(model, tokenizer, prompt: str, sampler) -> str:
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    return generate(
        model, tokenizer, prompt=prompt_text, max_tokens=150,
        sampler=sampler, verbose=False,
    ).strip()


def main() -> None:
    if not ADAPTER_PATH.exists():
        print(f"No adapter at {ADAPTER_PATH}. Run Module 4 LoRA training first.")
        raise SystemExit(1)

    prompts = [
        line.strip()
        for line in TEST_PROMPTS_FILE.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    print(f"Loaded {len(prompts)} held-out test prompts.\n")

    print(f"Loading BASE model: {BASE_MODEL}")
    base_model, base_tokenizer = load(BASE_MODEL)

    print(f"Loading FINE-TUNED model (base + adapter)…\n")
    ft_model, ft_tokenizer = load(BASE_MODEL, adapter_path=str(ADAPTER_PATH))

    sampler = make_sampler(temp=0.0)   # greedy → deterministic → fair comparison

    base_scores, ft_scores = [], []

    for i, prompt in enumerate(prompts, start=1):
        print(f"{'=' * 74}")
        print(f"Prompt {i}: {prompt}")
        print(f"{'=' * 74}")

        base_reply = reply(base_model, base_tokenizer, prompt, sampler)
        base_score = score_pirate_ness(base_reply)
        base_scores.append(base_score)
        print(f"\n[BASE  — pirate score = {base_score}]")
        print(base_reply)

        ft_reply = reply(ft_model, ft_tokenizer, prompt, sampler)
        ft_score = score_pirate_ness(ft_reply)
        ft_scores.append(ft_score)
        print(f"\n[TUNED — pirate score = {ft_score}]")
        print(ft_reply)
        print()

    base_avg = sum(base_scores) / len(base_scores)
    ft_avg = sum(ft_scores) / len(ft_scores)

    print(f"{'=' * 74}")
    print("SUMMARY")
    print(f"{'=' * 74}")
    print(f"  Test prompts:                    {len(prompts)}")
    print(f"  Base model    — avg pirate score: {base_avg:.2f} markers/response")
    print(f"  Fine-tuned    — avg pirate score: {ft_avg:.2f} markers/response")
    if base_avg > 0:
        print(f"  Ratio (higher = more style transfer):  {ft_avg / base_avg:.1f}x")
    else:
        print(f"  Ratio: base scored zero → fine-tuned is {ft_avg:.1f} markers vs 0.")
    print()
    print("Rule of thumb: 5x or higher ratio = strong style transfer.")


if __name__ == "__main__":
    main()
