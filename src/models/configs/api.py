"""Registers every API-provider model. These use the generic OpenAI-compatible
adapter (src/llm/openai_compatible_provider.py) instead of loading local
weights. Nothing "loads" at startup — the API key is only read (from the env
var named below) when you actually switch to one of these with /models <id>.
Point base_url/model_name at whichever OpenAI-wire-format provider you use
(OpenAI itself, Groq, Together, OpenRouter, a local vLLM/Ollama OpenAI-compat
server, etc.) and export the matching API key before switching.
"""

from src.models.registry import registry

registry.register(
    model_id="gpt-oss-120b-groq",
    name="GPT-OSS 120B (Groq API)",
    family="GPT-OSS",
    quantization="none",
    capabilities=["text"],
    provider="openai_compatible",
    context_length=131072,
    parameters="120B",
    license="Apache-2.0",
    source="https://console.groq.com/docs/models",
    description="OpenAI's open-weight GPT-OSS 120B served by Groq's OpenAI-compatible API. Requires GROQ_API_KEY.",
    provider_config={
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model_name": "openai/gpt-oss-120b",
    },
    rate_limits={
        "requests_per_minute": 30,
        "requests_per_day": 1000,
        "tokens_per_minute": 8000,
        "tokens_per_day": 200000,
    },
)

registry.register(
    model_id="llama-3.3-70b-groq",
    name="Llama 3.3 70B (Groq API)",
    family="Llama-3.3",
    quantization="none",
    capabilities=["text"],
    provider="openai_compatible",
    context_length=131072,
    parameters="70B",
    license="Llama 3.3 Community License",
    source="https://console.groq.com/docs/models",
    description="Meta's Llama 3.3 70B served by Groq's OpenAI-compatible API. Requires GROQ_API_KEY.",
    provider_config={
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model_name": "llama-3.3-70b-versatile",
    },
    rate_limits={
        "requests_per_minute": 30,
        "requests_per_day": 1000,
        "tokens_per_minute": 12000,
        "tokens_per_day": 100000,
    },
)

registry.register(
    model_id="qwen3-32b-groq",
    name="Qwen3 32B (Groq API)",
    family="Qwen3",
    quantization="none",
    capabilities=["text"],
    provider="openai_compatible",
    context_length=131072,
    parameters="32B",
    license="Apache-2.0",
    source="https://console.groq.com/docs/models",
    description="Alibaba's Qwen3 32B served by Groq's OpenAI-compatible API. Requires GROQ_API_KEY.",
    provider_config={
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model_name": "qwen/qwen3-32b",
    },
    rate_limits={
        "requests_per_minute": 60,
        "requests_per_day": 1000,
        "tokens_per_minute": 6000,
        "tokens_per_day": 500000,
    },
)
