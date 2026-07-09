# Configuring Models

Tuffy can run entirely offline on local model weights, or talk to any OpenAI-wire-format API
endpoint — both are registered the same way, as a "model card", and both show up together in
`/models`. Model cards are split by how the model is served:

- [src/models/configs/local.py](../src/models/configs/local.py) — local GGUF (`llama_cpp`) models.
- [src/models/configs/api.py](../src/models/configs/api.py) — API-provider (`openai_compatible`)
  models.

[src/models/__init__.py](../src/models/__init__.py) just imports both files (registration is a
side effect of the import) and sets `DEFAULT_MODEL`. See [src/llm/README.md](../src/llm/README.md)
for how the provider interface underneath this works.

## Adding a local model

1. Create `src/models/weights/<model_id>/` (pick `model_id` as
   `family-params-variant-quant`, all lowercase — fully spelled out so two quantizations of the
   same base model never collide).
2. Download the model's weight file (GGUF format, for the built-in `llama_cpp` provider) into
   that folder.
3. If it's a vision/audio/omni model, also download its matching projector file
   (`mmproj-*.gguf`) into the same folder — this is a separate file from the language model
   itself, usually in the same source repo.
4. Register it in [src/models/configs/local.py](../src/models/configs/local.py):

```python
from src.models.registry import registry

registry.register(
    model_id="my-model-7b-q4",
    name="My Model 7B (Q4)",
    family="my-model",
    quantization="Q4_K_M",
    path="src/models/weights/my-model-7b-q4/my-model.gguf",
    capabilities=["text"],           # add "vision" if you set mmproj_path
    context_length=8192,
    parameters="7B",
    license="...",
    source="...",                    # where the weights came from
    description="One line: what this model is and why you'd pick it.",
    load_params={...},               # overrides for llama_cpp.Llama.__init__ defaults
    sampling_params={...},           # overrides for create_chat_completion defaults
)
```

5. Run Tuffy, check `/models` lists it, then `/models <model_id>` to load and switch to it.

## Adding an API-provider model

Any endpoint that speaks the OpenAI chat-completions wire format works — no SDK dependency
beyond HTTP. Register it in [src/models/configs/api.py](../src/models/configs/api.py):

```python
registry.register(
    model_id="my-provider-model-name",   # end with the provider name, e.g. "-groq"
    name="My API Model (My Provider)",
    family="...",
    quantization="none",
    capabilities=["text"],           # add "vision" if the endpoint accepts image content blocks
    provider="openai_compatible",
    context_length=128000,           # the provider's documented max context window
    parameters=None,
    license=None,
    source="...",
    description="One line: what this is and which env var it needs.",
    provider_config={
        "base_url": "https://api.example.com/v1",
        "api_key_env": "MY_PROVIDER_API_KEY",
        "model_name": "provider-side-model-name",
    },
    rate_limits={                    # optional — shown in /status and the load banner
        "requests_per_minute": 30,
        "requests_per_day": 1000,
        "tokens_per_minute": 8000,
        "tokens_per_day": 200000,
    },
)
```

`rate_limits` is metadata only — nothing in this codebase enforces or throttles against it, it's
just displayed so you know your headroom. The provider's own API still returns `429` if you
exceed its real limits; see below for what happens when it does.

Then, before switching to it: `export MY_PROVIDER_API_KEY=...`, or add it to `.env` in the repo
root (`MY_PROVIDER_API_KEY=...`) — Tuffy loads `.env` automatically at startup via
[src/env.py](../src/env.py), filling in any variable not already exported in your shell. A real
exported env var always takes precedence over `.env`. See
[src/README.md](../src/README.md#environment-variables) for details. `.env` is gitignored, so the
key is never committed.

Switching with `/models my-provider-model-name` loads the new provider *before* unloading the
current one, so a bad switch (missing/invalid API key) leaves you on the previously working model
instead of with no agent at all.

### If an API request fails mid-turn

Rate limits, network errors, bad keys, and timeouts from an API-provider model surface as a
`ProviderError` (see [src/llm/base.py](../src/llm/base.py)), which the CLI's turn loop catches
the same way it handles a local out-of-memory decode failure: the failed turn is dropped, a
one-line `[Generation failed: ...]` message is printed, and the session stays alive — no crash,
no lost history from earlier turns. Just retry or switch models with `/models <id>`.

## Inspecting a model card

`/models info <id>` prints the full card: capabilities, context length, license, source, rate
limits (if declared), and (for API models) the base URL, model name, and whether the required
env var is currently set.

## Checking context/token usage

`/status` shows the active model's estimated context usage (current session history size vs. the
model's declared `context_length`) and its rate limits, if any. The token count is an estimate
(characters ÷ 4) — no provider here returns exact usage counts mid-stream, so treat it as a rough
gauge, not an exact figure.
