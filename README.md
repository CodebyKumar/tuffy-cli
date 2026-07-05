# Tuffy ⚡

Tuffy is a direct, capable personal AI agent that runs completely locally on your machine. It utilizes a local Qwen3-VL 2B model via `llama_cpp` to interact with you and execute real-world actions through Python-defined tools.

---

## Features

- **Local Execution**: Keep your data private. Everything is processed locally on your hardware.
- **Unified Chat Mode**: Displays a clean console interface featuring colored step-by-step traces showing thoughts, tool calls, and responses.
- **Dynamic Status Spinner**: Cycles through realistic thinking phases (`thinking`, `wandering`, `framing`, `processing`, etc.) with hidden terminal cursor and smooth dot sequences.
- **Synchronous Memory**: Fact extraction and session summary storage run synchronously on turn completions, avoiding background threads and memory spikes.
- **Local Model Optimization**: Built on `qwen3vl-2b-instruct-q4km` (4-bit quantization) with optimized parameters (`flash_attn` and smaller batch size) to run efficiently even on memory-constrained devices (like an 8GB Mac Air).

---

## Available Commands

When chatting with Tuffy, you can use these slash commands:
- `/memory` - Show everything in long-term memory (facts about you, recent sessions, lessons learned).
- `/clear` - Wipe long-term memory and conversation history.
- `/tools` - List all tools the agent can call and what they do.
- `/models` - List available models and show which one is active.
- `/image <path>` - Attach an image file to your next message (requires vision model).
- `/exit` or `/quit` - Save session memory and close the program.

---

## Setup & Running

### 1. Requirements
- Python 3.11+
- virtualenv (e.g. using `uv` or standard Python `venv`)

### 2. Installation
Set up your virtual environment and install the dependencies:
```bash
# Using uv (recommended)
uv sync

# Or using standard pip
pip install -r pyproject.toml
```

### 3. Model Weights
Place your model GGUF files in `src/models/weights/` (ignored by git). The default model expects:
- Language Model: `src/models/weights/qwen3vl-2b-instruct-q4km/Qwen3VL-2B-Instruct-Q4_K_M.gguf`
- CLIP Projector: `src/models/weights/qwen3vl-2b-instruct-q80/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf`

### 4. Running the Agent
Run the main script to start your local chat session:
```bash
python3 main.py
```
