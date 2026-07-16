"""
Hit the local mlx_lm.server via HTTP using the OpenAI SDK.

Assumes the server is running in another terminal:

    mlx_lm.server \\
      --model mlx-community/Qwen3-8B-4bit \\
      --adapter-path ./learn/04-finetune/adapters \\
      --port 8080

The whole point of an OpenAI-compatible server: the SAME code that hits
api.openai.com works here — only base_url changes.

Prereqs:
  - openai package (from Module 1)
  - mlx_lm.server running on port 8080

Run:
    python learn/05-evaluate-serve/client_test.py
"""
from __future__ import annotations

from openai import OpenAI


# Must match the --model argument you passed to `mlx_lm.server` in the other terminal.
# If you started the server with `--model ./learn/04-finetune/qwen3-pirate`, this must be
# the same string. Change it if you served the base+adapter combo instead.
MODEL_ID = "./learn/04-finetune/qwen3-pirate"

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-used",   # any string; mlx_lm.server ignores it
)


PROMPTS = [
    "Hello, who are you?",
    "What's the meaning of life?",
    "How can I learn to code?",
]


def main() -> None:
    print("Sending prompts to http://localhost:8080 (mlx_lm.server)…\n")

    for prompt in PROMPTS:
        print(f"USER: {prompt}")

        stream = client.chat.completions.create(
            # IMPORTANT: mlx_lm.server treats `model` as an identifier and will
            # try to load whatever repo/path you name. Pass the SAME string the
            # server was started with (--model flag) so it reuses the loaded model.
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=180,
        )

        print("ASSISTANT: ", end="", flush=True)
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    main()
