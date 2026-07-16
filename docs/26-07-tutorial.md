# AI Models — Hands-On Course (Phase 1)

> A practical, do-it-yourself path from "what is an AI model" to "I trained and fine-tuned my own," tuned for an **Apple M3 Pro / 18 GB / macOS Tahoe**.
> Written July 2026. Tools move fast — I'll re-verify versions when we actually run each step.

---

## How to use this course

- This file is the **course syllabus/overview**. The actual teaching content lands module-by-module in [learn/knowledge.md](../learn/knowledge.md) as you request each module; code and scripts live under `learn/`.
- Each chapter has: **Goal → Concepts → Hands-on → Checkpoint.**
- Don't skip the hands-on. The whole point is muscle memory, not theory.
- A "Checkpoint" is how you know you're ready for the next chapter. Tell me when you hit one and we'll do it together.
- **Pacing rule:** we move strictly one module at a time. We advance only when you explicitly name the next module (e.g. "move on to Module 2"). Until then, we stay put — questions, re-explanations, and extra practice welcome.
- **Assumed background:** zero AI/ML knowledge; moderate software/engineering skills (terminal, git, running code).
- Estimated total: **4–7 weeks** at a few hours per week. No rush — depth over speed.

### The 6 modules at a glance

| # | Module | What you'll be able to do | Est. |
|---|--------|---------------------------|------|
| 0 | Setup & mental model | Understand what a model *is*; get the Mac ready | 2–3 hrs |
| 1 | Run models locally | Chat with local LLMs; use them from Python | 3–5 hrs |
| 2 | How models work inside | Explain tokens, embeddings, transformers, inference | 4–6 hrs |
| 3 | Train a tiny model from scratch | Build & train a small neural net + a mini-GPT | 8–12 hrs |
| 4 | Fine-tune a real model | LoRA fine-tune a real LLM on your own data (MLX) | 6–10 hrs |
| 5 | Evaluate, quantize & serve | Measure quality, shrink models, serve an API | 4–6 hrs |
| 6 | Bridge to Phase 2 | Understand EEG/time-series models for the BCI project | 3–4 hrs |

---

## Module 0 — Setup & the mental model

**Goal:** Know what an "AI model" actually is, and have a working ML environment on the Mac.

### Concepts

- **A model = weights + architecture.** The *architecture* is the wiring (how numbers flow). The *weights* are millions/billions of numbers learned from data. "Running a model" = pushing your input through that wiring using those numbers.
- **Training vs inference.** *Training* = adjusting the weights so predictions improve (expensive, done once). *Inference* = using frozen weights to get an answer (cheap, done constantly).
- **Why Apple Silicon is special:** CPU and GPU share **one pool of memory** (unified memory). That's why an 18 GB Mac can run models that would need an expensive discrete GPU elsewhere. Apple's **MLX** framework is built specifically for this.
- **Parameters & memory rule of thumb:** at 4-bit quantization, a model needs roughly *(billions of params) × 0.5–0.6 GB*. So a 7B model ≈ ~4–5 GB, a 14B ≈ ~8–9 GB. Add 1–2 GB for context. On 18 GB you have real room up to ~14B, and can reach ~30B MoE models.

### Hands-on

1. Install **Homebrew** (Mac package manager) if not present:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. Install Python tooling. We'll use `uv` (fast, modern) or plain `venv`:
   ```bash
   brew install python@3.12 uv git
   ```
3. Create a project env:
   ```bash
   cd ~/destress
   uv venv .venv && source .venv/bin/activate
   uv pip install numpy torch mlx-lm jupyter
   ```
4. Confirm PyTorch sees the Mac GPU (Metal / **MPS**):
   ```bash
   python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
   ```

### Checkpoint

- [ ] I can explain "weights vs architecture" and "training vs inference" in my own words.
- [ ] `MPS available: True` prints on my machine.

---

## Module 1 — Run models locally

**Goal:** Chat with local LLMs three ways (GUI, CLI, code) and know when to use each.

### Concepts

- **Three tools, three jobs (2026 landscape):**
  - **LM Studio** — friendly GUI. Best for browsing/downloading models and quick chat. Uses MLX-optimized models on Apple Silicon.
  - **Ollama** — one-line CLI + local API server. Modern versions use Apple's **MLX** backend under the hood, giving a big speed-up over the old llama.cpp path. Best for scripting and app development.
  - **MLX-LM** — Apple's native library. Maximum performance and the tool we'll use for fine-tuning later.
- **Quantization** = storing weights in fewer bits (16-bit → 8/4-bit). Smaller + faster, tiny quality loss. `Q4` / `4-bit` is the sweet spot on 18 GB.
- **Good models for 18 GB (July 2026):** the **Qwen3** family (1.7B → 14B, Apache-2.0) is the safe default; **Gemma** and **Llama** small variants are strong alternatives. Start with an 8B-class model.

### Hands-on

1. **GUI path:** download LM Studio (lmstudio.ai), grab an MLX build of an 8B model (e.g. Qwen3-8B), chat with it.
2. **CLI path:**
   ```bash
   brew install ollama
   ollama serve   # starts the local server (leave running)
   ollama run qwen3:8b   # first run downloads, then chats
   ```
3. **Code path** — talk to the local server from Python (OpenAI-compatible API):
   ```python
   from openai import OpenAI
   client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
   r = client.chat.completions.create(
       model="qwen3:8b",
       messages=[{"role": "user", "content": "Explain LoRA in 2 sentences."}],
   )
   print(r.choices[0].message.content)
   ```
4. Watch memory + speed: run `ollama ps` and note tokens/sec. Try a 4B vs 14B model and feel the trade-off.

### Checkpoint

- [ ] I've chatted with a local model via GUI, CLI, and Python.
- [ ] I can pick a model size for 18 GB and say why.

---

## Module 2 — How models work inside

**Goal:** Open the black box. Explain, at a whiteboard level, how text becomes a prediction.

### Concepts (the LLM pipeline)

```
Text ──▶ Tokenizer ──▶ Token IDs ──▶ Embeddings ──▶ Transformer layers ──▶ Logits ──▶ Sampler ──▶ Next token
                                          ▲                                                            │
                                          └──────────────── repeat (autoregressive) ◀─────────────────┘
```

- **Tokens:** models don't see words, they see chunks ("token" ≈ ~¾ of a word). The **tokenizer** maps text ↔ integer IDs (BPE = byte-pair encoding).
- **Embeddings:** each token ID becomes a vector (a list of numbers) that captures meaning. Similar meanings → nearby vectors.
- **Transformer + attention:** the core trick. **Attention** lets each token "look at" other tokens to build context ("it" → which noun?). Layers stack this to build understanding.
- **Logits → sampling:** the model outputs a score for every possible next token. **Temperature / top-p** control how randomly we pick. Then it appends the token and repeats — this is **autoregression**.
- **Context window & KV cache:** how many tokens it can attend to at once, and the memory that grows with conversation length.

### Hands-on

1. **See tokenization:**
   ```python
   from transformers import AutoTokenizer
   tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
   ids = tok.encode("Stress detection with EEG is fun!")
   print(ids); print([tok.decode([i]) for i in ids])
   ```
2. **Watch it generate one token at a time** — set `max_tokens=1` in the Module 1 code, feed the output back in a loop, and print each step.
3. **Feel temperature:** ask the same question at `temperature=0` vs `1.2` and compare.
4. **Watch (recommended):** Andrej Karpathy's *"Let's build GPT"* and the *Neural Networks: Zero to Hero* series — the clearest explanation of everything above.

### Checkpoint

- [ ] I can draw the pipeline above from memory and explain each box.
- [ ] I understand why longer chats use more memory.

---

## Module 3 — Train a tiny model *from scratch*

**Goal:** Demystify training by building and training real (small) models yourself. This is the heart of the course.

### Concepts

- **The training loop (the single most important idea):**
  ```
  forward pass → compute loss → backward pass (gradients) → optimizer step → repeat
  ```
  - **Loss** = how wrong the prediction is (one number).
  - **Backprop** = calculus that tells each weight which direction to move.
  - **Optimizer (e.g. Adam)** = nudges weights to reduce loss.
  - **Epoch** = one pass over the data.
- **Overfitting vs generalisation:** memorising the training data vs actually learning. We watch a *validation* loss to catch it.

### Hands-on (three escalating projects)

1. **Micrograd (understand backprop):** follow Karpathy's `micrograd` — build a tiny autograd engine by hand. ~150 lines. You'll *see* gradients flow.
2. **Your first PyTorch net on MPS:** train a small classifier (e.g. MNIST digits) on the Mac GPU:
   ```python
   import torch, torch.nn as nn
   device = "mps"
   model = nn.Sequential(nn.Flatten(), nn.Linear(784,128), nn.ReLU(), nn.Linear(128,10)).to(device)
   # ... standard train loop: forward → loss → backward → step
   ```
   Goal: watch loss go down and accuracy go up — that's training, live.
3. **nanoGPT (train a mini language model):** clone Karpathy's `nanoGPT`, train a character-level GPT on a small text file (e.g. Shakespeare) with `--device=mps`. In minutes you'll have a model that generates text in that style. This is a *real* transformer, just small.
   ```bash
   git clone https://github.com/karpathy/nanoGPT && cd nanoGPT
   python data/shakespeare_char/prepare.py
   python train.py config/train_shakespeare_char.py --device=mps --compile=False
   python sample.py --out_dir=out-shakespeare-char --device=mps
   ```

### Checkpoint

- [ ] I trained a net on MPS and watched loss drop.
- [ ] My nanoGPT generates (bad but real) text — and I understand every stage that produced it.

---

## Module 4 — Fine-tune a *real* model on your own data

**Goal:** Take a pretrained LLM and teach it your task/style with **LoRA** on the Mac — cheaply, locally.

### Concepts

- **Why not train from scratch?** Pretraining a real LLM costs millions. Instead we **fine-tune**: start from open weights and adjust slightly.
- **LoRA (Low-Rank Adaptation):** freeze the huge original weights, train only small "adapter" matrices bolted on. Result: fast, low-memory, and you get a tiny adapter file (MBs) instead of a whole new model. Perfect for 18 GB.
- **QLoRA:** LoRA *on top of a quantized* model → even less memory. MLX supports training adapters over 4-bit models directly.
- **Data format:** usually JSONL of prompt/response (or chat) examples. Quality > quantity; even a few hundred good examples help.

### Hands-on

1. Install and prep:
   ```bash
   uv pip install mlx-lm
   ```
2. Make a `data/` folder with `train.jsonl` / `valid.jsonl` (chat-format examples of the behavior you want).
3. Fine-tune with one command:
   ```bash
   mlx_lm.lora \
     --model mlx-community/Qwen3-8B-4bit \
     --train \
     --data ./data \
     --iters 600 \
     --batch-size 1
   ```
4. Test the adapter, then **fuse** it into a standalone model if you like:
   ```bash
   mlx_lm.generate --model mlx-community/Qwen3-8B-4bit --adapter-path ./adapters --prompt "..."
   mlx_lm.fuse --model mlx-community/Qwen3-8B-4bit --adapter-path ./adapters
   ```

> Note: A small LoRA run finishes in tens of minutes on Apple Silicon for $0 cloud cost. This is the exact technique we'll reuse in Phase 2 if we fine-tune a model on EEG-derived features.

### Checkpoint

- [ ] I fine-tuned a real model on my own examples and saw its behavior change.
- [ ] I can explain LoRA and why it fits an 18 GB machine.

---

## Module 5 — Evaluate, quantize & serve

**Goal:** Turn a model into something you can trust and use like a real service.

### Concepts

- **Evaluation:** don't trust "it looks good." Use held-out test data; measure accuracy/perplexity for LLMs, or task metrics (precision/recall/F1) for classifiers — the ones we'll use in Phase 2.
- **Quantization (revisited):** convert your fine-tuned model to 4/8-bit to run faster and smaller. `mlx_lm.convert -q`.
- **Serving:** expose the model over an HTTP API so an app can call it.
  ```bash
  mlx_lm.server --model ./your-model   # local OpenAI-compatible endpoint
  ```
- **Prompting vs fine-tuning vs RAG:** know which problem each solves so you don't fine-tune when a better prompt would do.

### Hands-on

1. Write a tiny eval script: loop over test prompts, compare to expected, print a score.
2. Quantize your Module 4 model and compare speed + size before/after.
3. Serve it and hit it from the Module 1 Python client.

### Checkpoint

- [ ] I can measure whether a model actually got better.
- [ ] I can serve a model locally and call it over HTTP.

---

## Module 6 — Bridge to Phase 2 (EEG / BCI)

**Goal:** Connect everything learned to the real project — stress detection from brain signals. (We plan this properly *after* you're confident on Modules 0–5.)

### Concepts (preview only)

- **Different data, same playbook.** EEG is **time-series** signal data, not text. But the loop is identical: data → model → train → evaluate → serve.
- **The likely pipeline:**
  ```
  BCI headset ─▶ raw EEG (µV over time) ─▶ preprocess & filter ─▶ features
        (bands: theta/alpha/beta) ─▶ ML model ─▶ stress score 0–100 ─▶ remedial action
  ```
- **Key EEG ideas we'll learn then:** frequency bands (theta rises with cognitive load; alpha shifts with attention), windowing, and why models like small CNNs / 1D-temporal nets or classic classifiers (SVM/gradient-boosting) are common for consumer-grade EEG.
- **Hardware options** we'll compare: **Muse** (4 forehead electrodes, easy) vs **OpenBCI** (Ganglion/Cyton, more channels, more control).
- **The mapping layer:** stress-score ranges → actions (deep breathing, calming music, favorite photos) is a simple rules table sitting on top of the model output.

### Checkpoint

- [ ] I can see how a training loop I built in Module 3 maps onto EEG data.
- [ ] I'm ready to design Phase 2 with you.

---

## Glossary (quick reference)

| Term | Plain meaning |
|------|---------------|
| Weights / parameters | The learned numbers that make a model work |
| Inference | Using a trained model to get an answer |
| Token | A chunk of text (~¾ word) the model actually processes |
| Embedding | A token turned into a vector of numbers (its "meaning") |
| Transformer / attention | The architecture that lets tokens use context |
| Quantization | Storing weights in fewer bits to save memory/speed |
| LoRA / QLoRA | Cheap fine-tuning by training small add-on adapters |
| MLX | Apple's ML framework, built for Apple Silicon unified memory |
| MPS | Metal Performance Shaders — PyTorch's Apple-GPU backend |
| Epoch / loss / backprop | One data pass / how wrong / how weights learn to improve |

---

## Curated resources

- **Karpathy — Neural Networks: Zero to Hero** (micrograd, makemore, nanoGPT). The single best hands-on foundation.
- **nanoGPT** and **build-nanogpt** repos — train a real GPT from scratch on your Mac.
- **MLX-LM** docs — running & fine-tuning on Apple Silicon.
- **Ollama** & **LM Studio** — local model running.
- **Hugging Face** — models, datasets, and the `transformers` library.

*(Exact links, versions, and model names to be re-verified live when we run each module — tooling changes monthly.)*
