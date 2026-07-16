"""
Chat with a local model served by Ollama, via its OpenAI-compatible HTTP API.

Streams the response so it feels like a real chat UI.

Prerequisites:
  - Ollama running (installed via `brew install ollama`; runs at startup, or
    manually via `ollama serve`)
  - Model pulled:  `ollama pull qwen3:8b`
  - Python `openai` package:  `uv pip install openai`

Run:
    python learn/01-run-local/chat_ollama.py
"""
from __future__ import annotations

from openai import OpenAI

MODEL = "qwen3:8b"   # try "gemma3:12b" or another tag; see `ollama list`

PROMPT = "Explain LoRA in 2 sentences, aimed at a software engineer new to ML."


def main() -> None:
    # Ollama's HTTP server on localhost:11434 speaks the OpenAI protocol.
    # Any OpenAI-SDK-compatible client works — swap base_url + api_key for a
    # cloud API and the rest of the code is identical.
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required by the SDK; Ollama ignores the value
    )

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a concise, friendly assistant."},
            {"role": "user", "content": PROMPT},
        ],
        stream=True,
    )

    for chunk in stream:
        piece = chunk.choices[0].delta.content
        if piece:
            print(piece, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
