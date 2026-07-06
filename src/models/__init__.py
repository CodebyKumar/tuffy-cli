"""Registers every model Tuffy can load. Add a new model by calling
registry.register(...) here with its full model card - it becomes available
to /models automatically, no other wiring needed.
"""

from src.models.registry import registry

DEFAULT_MODEL = "qwen3vl-2b-instruct-q4km"

registry.register(
    model_id="qwen3vl-2b-instruct-q4km",
    name="Qwen3-VL 2B Instruct (Q4_K_M)",
    family="Qwen3-VL",
    quantization="Q4_K_M",
    path="src/models/weights/qwen3vl-2b-instruct-q4km/Qwen3VL-2B-Instruct-Q4_K_M.gguf",
    capabilities=["text", "vision"],
    # The vision projector (mmproj) is independent of the language model's
    # quantization, so both Qwen3-VL variants share the one Q8_0 mmproj file
    # already downloaded into the q80 subfolder.
    mmproj_path="src/models/weights/qwen3vl-2b-instruct-q80/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf",
    context_length=4096,
    parameters="2B",
    license="Apache-2.0",
    source="Qwen/Qwen3-VL-2B-Instruct",
    description="Small instruction-tuned vision-language model; used as Tuffy's default local model.",
    load_params={
        "n_ctx": 4096,
        # Keep n_batch small: bigger batches inflate the Metal compute buffer
        # and starve an 8GB machine into llama_decode -3 failures. The mtmd
        # helper chunks image embeddings, so images work fine at 512.
        "n_batch": 512,
        "n_ubatch": 512,
        "n_threads": 4,
        "n_gpu_layers": -1,
        # Shrinks the Metal compute buffer enough that the LLM and the GPU
        # vision encoder fit together on an 8GB machine.
        "flash_attn": True,
    },
    sampling_params={
        # Deterministic (0.0) made the model give the identical canned reply
        # every time a question repeated; a little temperature plus repeat
        # penalty keeps answers fresh without derailing tool-call syntax.
        "temperature": 0.3,
        "repeat_penalty": 1.1,
    },
)

# --- API-provider models -----------------------------------------------
# These use the generic OpenAI-compatible adapter (src/llm/openai_compatible_provider.py)
# instead of loading local weights. Nothing "loads" at startup — the API key
# is only read (from the env var named below) when you actually switch to
# one of these with /models <id>. Point base_url/model_name at whichever
# OpenAI-wire-format provider you use (OpenAI itself, Groq, Together,
# OpenRouter, a local vLLM/Ollama OpenAI-compat server, etc.) and export the
# matching API key before switching.

registry.register(
    model_id="gpt-4o-mini-api",
    name="GPT-4o mini (API)",
    family="GPT-4o",
    quantization="none",
    capabilities=["text", "vision"],
    provider="openai_compatible",
    context_length=128000,
    parameters=None,
    license=None,
    source="https://platform.openai.com/docs/models",
    description="OpenAI's GPT-4o mini via the OpenAI API. Requires OPENAI_API_KEY.",
    provider_config={
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model_name": "gpt-4o-mini",
    },
)

registry.register(
    model_id="llama-3.3-70b-groq-api",
    name="Llama 3.3 70B (Groq API)",
    family="Llama-3.3",
    quantization="none",
    capabilities=["text"],
    provider="openai_compatible",
    context_length=128000,
    parameters="70B",
    license="Llama 3.3 Community License",
    source="https://console.groq.com/docs/models",
    description="Meta's Llama 3.3 70B served by Groq's OpenAI-compatible API. Requires GROQ_API_KEY.",
    provider_config={
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model_name": "llama-3.3-70b-versatile",
    },
)

registry.register(
    model_id="qwen3vl-2b-instruct-q80",
    name="Qwen3-VL 2B Instruct (Q8_0)",
    family="Qwen3-VL",
    quantization="Q8_0",
    path="src/models/weights/qwen3vl-2b-instruct-q80/Qwen3VL-2B-Instruct-Q8_0.gguf",
    capabilities=["text", "vision"],
    mmproj_path="src/models/weights/qwen3vl-2b-instruct-q80/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf",
    context_length=4096,
    parameters="2B",
    license="Apache-2.0",
    source="Qwen/Qwen3-VL-2B-Instruct",
    description="Small instruction-tuned vision-language model; used as Tuffy's default local model.",
    load_params={
        "n_ctx": 4096,
        # Keep n_batch small: bigger batches inflate the Metal compute buffer
        # and starve an 8GB machine into llama_decode -3 failures. The mtmd
        # helper chunks image embeddings, so images work fine at 512.
        "n_batch": 512,
        "n_ubatch": 512,
        "n_threads": 4,
        "n_gpu_layers": -1,
        # Shrinks the Metal compute buffer enough that the LLM and the GPU
        # vision encoder fit together on an 8GB machine.
        "flash_attn": True,
    },
    sampling_params={
        # Deterministic (0.0) made the model give the identical canned reply
        # every time a question repeated; a little temperature plus repeat
        # penalty keeps answers fresh without derailing tool-call syntax.
        "temperature": 0.3,
        "repeat_penalty": 1.1,
    },
)
