# src/llm/

LLM provider adapters. `src/agent.py`'s ReAct loop never talks to `llama_cpp` or an HTTP client
directly — it only ever calls `provider.stream_completion(...)` / `provider.complete(...)`
through the interface defined here, so swapping or adding a backend never touches the agent loop.

- [base.py](base.py) - `LLMProvider`, the abstract interface: `load()`/`unload()` lifecycle
  hooks and `complete()`/`stream_completion()`, both shaped like llama.cpp's own
  `create_chat_completion` response format (the lowest common denominator every adapter
  normalizes its own wire format to).
- [llama_cpp_provider.py](llama_cpp_provider.py) - `LlamaCppProvider`: local GGUF models via
  `llama_cpp.Llama`, including the MTMD vision handler for multimodal models.
- [openai_compatible_provider.py](openai_compatible_provider.py) - `OpenAICompatibleProvider`:
  any OpenAI-wire-format HTTP endpoint (OpenAI, Groq, Together, OpenRouter, a local vLLM/Ollama
  server, ...) via `base_url` + `api_key_env` + `model_name` from the model card.
- [__init__.py](__init__.py) - `PROVIDERS` (provider name → adapter class) and
  `build_provider(model_card)`, which [src/agent.py](../agent.py) calls once per model load.

## Adding a new provider

1. Write `src/llm/<name>_provider.py` implementing `LLMProvider`'s four methods
   (`load`, `unload`, `complete`, `stream_completion`).
2. Register it in `__init__.py`'s `PROVIDERS` dict under whatever string you want model cards
   to use for `provider`.
3. Add a model card in [src/models/configs/api.py](../models/configs/api.py) (or a new
   `configs/<name>.py`, imported from [models/__init__.py](../models/__init__.py)) with that
   `provider` value and a `provider_config` dict of whatever your adapter's `__init__` needs.

No changes to `src/agent.py`, `main.py`, or the tool/prompt/memory systems are ever required —
they only see the model card's `provider` field and this interface.

## API keys

Read from environment variables only, named by the model card's `api_key_env` — never written
to a file in the repo, never logged. That variable can be exported in your shell, or placed in
a `.env` file at the repo root (gitignored); [src/env.py](../env.py) loads `.env` into
`os.environ` at startup for anything not already exported, so a real exported var always wins.
See [src/models/configs/api.py](../models/configs/api.py) for the env vars the registered API
model cards expect.

A failed API call (rate limit, bad key, network error, timeout) raises `ProviderError`
(defined in [base.py](base.py)) rather than a raw `requests` exception, so
[src/cli/turn.py](../cli/turn.py) can catch it uniformly alongside local decode failures and
keep the session alive instead of crashing.
