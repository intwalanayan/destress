# destress

> A hands-on, zero-to-hero AI/ML learning project for Apple Silicon Macs — followed by a real-world Brain-Computer Interface stress detector built on the same foundations.

## Highlights

- **Six-module curriculum** from *"what is a model?"* to *"I fine-tuned and served my own LLM"*
- **Runs entirely on Apple Silicon** (M-series) with 16 GB+ unified memory — no cloud GPU needed
- **Every concept paired with runnable code** you write and execute yourself
- **Trains models from scratch** (nanoGPT-style) and **fine-tunes Qwen3-8B with LoRA**
- **Phase 2 (coming)**: EEG stress detection via a consumer BCI kit + a small classifier
- **Built collaboratively with Claude Code** — the AI-agent workflow is documented in [`AGENTS.md`](AGENTS.md)

## What is this?

**destress** is a two-phase project:

1. **Phase 1 (current)** — Learn AI models hands-on. Six modules, ~4–6 weeks part-time, culminating in a fine-tuned Qwen3-8B LLM served as an HTTP API on your Mac. Zero cloud spend.
2. **Phase 2 (coming later)** — Build a real Brain-Computer Interface stress detector. Capture EEG signals from a consumer BCI kit (Muse or OpenBCI), feed them to a small classifier trained on your own recordings, output a stress score (0–100) plus mapped remedial actions (deep breathing, calming music, favorite photos, etc.).

Phase 2 begins once Phase 1's foundations feel solid.

## Who is this for?

- Software engineers curious about AI/ML but new to the field
- Anyone with a Mac who wants to touch real LLMs (not just call an API)
- Learners who prefer worked examples over lectures
- Anyone planning a real applied-ML project and wants the foundations first

## Requirements

- Mac with **Apple Silicon** (M-series chip)
- **16 GB+ unified memory** (18 GB tested; 8 GB is very tight)
- macOS 14+ (tested on Tahoe 26.5.1)
- ~15 GB free disk (models cache locally)
- Comfort with the terminal, `git`, and running Python code

## Quick start

Clone the repo:

```bash
git clone https://github.com/intwalanayan/destress.git
cd destress
```

Install prerequisites (skip whatever you already have):

```bash
brew install python@3.12 uv git
```

Create a Python 3.12 virtual environment and install the ML stack:

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install numpy torch mlx mlx-lm jupyter openai transformers torchvision
```

Verify your Apple GPU is accessible:

```bash
python learn/00-setup/verify_environment.py
```

Expected: every check prints `[PASS]`.

Open the course material:

```bash
open docs/2026-07-tutorial.md    # course syllabus
open learn/knowledge.md          # full teaching content, all modules
```

Start with Module 0. Work through each module in order — each module has a `Concepts` section (theory + diagrams) and a `Hands-on` section (runnable scripts).

## Phase 1 modules at a glance

| # | Module | What you'll be able to do | Est. |
|---|--------|---------------------------|------|
| 0 | Setup & mental model | Understand what a model *is*; get the Mac ready | 2–3 hrs |
| 1 | Run models locally | Chat with local LLMs; use them from Python | 3–5 hrs |
| 2 | How models work inside | Explain tokens, embeddings, transformers, inference | 4–6 hrs |
| 3 | Train a tiny model from scratch | Build & train a small neural net + a mini-GPT | 8–12 hrs |
| 4 | Fine-tune a real model | LoRA fine-tune Qwen3-8B on your own data (MLX) | 6–10 hrs |
| 5 | Evaluate, quantize & serve | Measure quality, compress, serve as an HTTP API | 4–6 hrs |
| 6 | Bridge to Phase 2 | EEG/time-series model foundations for BCI project | 3–4 hrs |

Full syllabus: [`docs/2026-07-tutorial.md`](docs/2026-07-tutorial.md). Complete teaching content: [`learn/knowledge.md`](learn/knowledge.md).

## Repository layout

```
destress/
├── AGENTS.md                    Instructions for AI agents assisting the learner
├── CLAUDE.md                    Claude Code entry point (imports AGENTS.md)
├── README.md                    This file
├── docs/
│   └── 2026-07-tutorial.md      Phase 1 syllabus / course overview
├── learn/
│   ├── knowledge.md             Full teaching content — all modules, appended by the AI agent
│   ├── 00-setup/                Module 0: environment verification + first GPU demos
│   ├── 01-run-local/            Module 1: chat via Ollama / MLX-LM + speed comparison
│   ├── 02-model-internals/      Module 2: tokenizer, one-token-at-a-time, temperature, logits
│   ├── 03-train/                Module 3: micrograd, MNIST classifier
│   ├── 04-finetune/             Module 4: LoRA fine-tuning Qwen3-8B (pirate persona demo)
│   └── 05-evaluate-serve/       Module 5: held-out evaluation + HTTP serving
├── references/
│   └── images/                  SVG diagrams used throughout the course
└── .claude/
    └── settings.json            Shared Claude Code project config
```

## How this was built

destress is a learning-by-doing project developed with the assistance of **Claude Code** — an AI-agent workflow. The rules for how the assistant collaborates with the learner (example-first pedagogy, plain-English-before-jargon, always answering *"who decides X and how?"*) are documented in [`AGENTS.md`](AGENTS.md).

If you fork this repo to teach yourself, an AI assistant reading `AGENTS.md` will pick up the same teaching style. Adapt the rules there to your own learning preferences.

## Roadmap

- [x] Module 0 — Setup & mental model
- [x] Module 1 — Run models locally
- [x] Module 2 — How models work inside
- [x] Module 3 — Train a tiny model from scratch
- [x] Module 4 — Fine-tune Qwen3-8B with LoRA
- [x] Module 5 — Evaluate, quantize & serve
- [ ] Module 6 — Bridge to Phase 2
- [ ] Phase 2 — EEG stress detector on real BCI hardware

## License

**MIT License** — see [`LICENSE`](LICENSE). Free to use, modify, and redistribute (including commercially); attribution required.

## Credits & references

Built on the shoulders of:

- [Andrej Karpathy — Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) — the canonical hands-on ML curriculum this project draws from heavily
- [nanoGPT](https://github.com/karpathy/nanoGPT) — used directly in Module 3 to train a real transformer on Shakespeare
- [Apple MLX](https://github.com/ml-explore/mlx) and [MLX-LM](https://github.com/ml-explore/mlx-lm) — the framework that makes Apple-Silicon fine-tuning practical
- [Ollama](https://ollama.com), [LM Studio](https://lmstudio.ai), [Hugging Face](https://huggingface.co) — the local-LLM ecosystem
- Qwen team at Alibaba — for [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), the model we fine-tune in Module 4

README structure informed by [README best practices](https://github.com/jehna/readme-best-practices) and the [awesome-readme](https://github.com/matiassingers/awesome-readme) collection.
