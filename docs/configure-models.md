# Configuring Models

Tuffy can run entirely offline on local model weights, or talk to any OpenAI-wire-format API
endpoint — both are registered the same way, as a "model card" in
[src/models/__init__.py](../src/models/__init__.py), and both show up together in `/models`.
See [src/llm/README.md](../src/llm/README.md) for how the provider interface underneath this
works.

## Adding a local model

1. Create `src/models/weights/<model_id>/` (pick `model_id` as
   `family-params-variant-quant`, all lowercase — fully spelled out so two quantizations of the
   same base model never collide).
2. Download the model's weight file (GGUF format, for the built-in `llama_cpp` provider) into
   that folder.
3. If it's a vision/audio/omni model, also download its matching projector file
   (`mmproj-*.gguf`) into the same folder — this is a separate file from the language model
   itself, usually in the same source repo.
4. Register it:

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
beyond HTTP.

```python
registry.register(
    model_id="my-api-model",
    name="My API Model",
    family="...",
    quantization="none",
    capabilities=["text"],           # add "vision" if the endpoint accepts image content blocks
    provider="openai_compatible",
    context_length=128000,
    parameters=None,
    license=None,
    source="...",
    description="One line: what this is and which env var it needs.",
    provider_config={
        "base_url": "https://api.example.com/v1",
        "api_key_env": "MY_PROVIDER_API_KEY",
        "model_name": "provider-side-model-name",
    },
)
```

Then, before switching to it: `export MY_PROVIDER_API_KEY=...`. The key is read from the
environment only at load time — never written to a file in the repo, never logged.

Switching with `/models my-api-model` loads the new provider *before* unloading the current one,
so a bad switch (missing/invalid API key) leaves you on the previously working model instead of
with no agent at all.

## Inspecting a model card

`/models info <id>` prints the full card: capabilities, context length, license, source, and
(for API models) the base URL, model name, and whether the required env var is currently set.
