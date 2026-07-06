"""Model registration infrastructure, mirroring src/registry.py's pattern for
tools: register() bookkeeps each model's full model-card metadata plus its
llama.cpp load params and sampling params, giving every field a sane default
that a dev can override per model. /models in main.py reads this registry to
list and switch models.
"""

# Every keyword argument llama_cpp.Llama.__init__ accepts, with the same
# defaults llama-cpp-python itself uses. A model's "load_params" dict below
# is merged over these, so a model only needs to specify what it overrides.
LOAD_PARAM_DEFAULTS = {
    "n_gpu_layers": 0,
    "split_mode": 1,
    "main_gpu": 0,
    "tensor_split": None,
    "vocab_only": False,
    "use_mmap": True,
    "use_mlock": False,
    "kv_overrides": None,
    "seed": 4294967295,
    "n_ctx": 512,
    "n_batch": 512,
    "n_ubatch": 512,
    "n_threads": None,
    "n_threads_batch": None,
    "rope_scaling_type": -1,
    "pooling_type": -1,
    "attention_type": -1,
    "rope_freq_base": 0.0,
    "rope_freq_scale": 0.0,
    "yarn_ext_factor": -1.0,
    "yarn_attn_factor": 1.0,
    "yarn_beta_fast": 32.0,
    "yarn_beta_slow": 1.0,
    "yarn_orig_ctx": 0,
    "logits_all": False,
    "embedding": False,
    "offload_kqv": True,
    "flash_attn": False,
    "op_offload": None,
    "swa_full": None,
    "no_perf": False,
    "last_n_tokens_size": 64,
    "lora_base": None,
    "lora_scale": 1.0,
    "lora_path": None,
    "numa": False,
    "chat_format": None,
    "chat_handler": None,
    "draft_model": None,
    "tokenizer": None,
    "type_k": None,
    "type_v": None,
    "spm_infill": False,
    "verbose": False,
}

# Every keyword argument llama_cpp.Llama.create_chat_completion accepts that
# controls sampling/generation behavior, with its own library defaults. A
# model's "sampling_params" dict below is merged over these.
SAMPLING_PARAM_DEFAULTS = {
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 40,
    "min_p": 0.05,
    "typical_p": 1.0,
    "stop": [],
    "seed": None,
    "max_tokens": None,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repeat_penalty": 1.0,
    "tfs_z": 1.0,
    "mirostat_mode": 0,
    "mirostat_tau": 5.0,
    "mirostat_eta": 0.1,
    "logit_bias": None,
}

# Recognized capability flags a model card can declare in "capabilities".
# text is implicit for every model; the rest describe extra modalities.
KNOWN_CAPABILITIES = {"text", "vision", "audio", "omni"}

# Which src/llm/*_provider.py adapter loads a card whose "provider" field
# matches. "llama_cpp" is a local gguf model (today's only kind);
# "openai_compatible" is any OpenAI-wire-format HTTP API (OpenAI itself,
# Groq, Together, OpenRouter, a local vLLM/Ollama OpenAI-compat server, etc).
KNOWN_PROVIDERS = {"llama_cpp", "openai_compatible"}


class ModelRegistry:
    def __init__(self):
        self.models = {}

    def register(
        self,
        model_id: str,
        name: str,
        family: str,
        quantization: str,
        capabilities: list[str] = ("text",),
        provider: str = "llama_cpp",
        path: str = None,
        mmproj_path: str = None,
        context_length: int = None,
        parameters: str = None,
        license: str = None,
        source: str = None,
        description: str = "",
        load_params: dict = None,
        sampling_params: dict = None,
        provider_config: dict = None,
    ) -> None:
        """Adds one model to the registry with its full model card. Works
        uniformly for local gguf models and API-provider models — /models and
        the switch logic in main.py never branch on this; only the src/llm/
        adapter picked by 'provider' cares which fields it reads.

        model_id: unique key used with /models <id>, e.g. 'qwen3vl-2b-instruct-q4km'
            or 'gpt-4o-mini-api'. Kept fully qualified (family + size + variant +
            quant, or family + '-api') so distinct variants never collide.
        name: human-readable full model card name, e.g.
            'Qwen3-VL 2B Instruct (Q4_K_M)'.
        family: base model family, e.g. 'Qwen3-VL' or 'GPT-4o'.
        quantization: quantization scheme, e.g. 'Q4_K_M'; 'none' for a full-precision
            local model or an API model where quantization isn't user-visible.
        capabilities: subset of KNOWN_CAPABILITIES this model supports. 'omni'
            means the model natively handles combined text/audio/vision I/O.
        provider: one of KNOWN_PROVIDERS — which src/llm/ adapter loads this card.
        path: (llama_cpp only) filesystem path to the model's own subfolder under
            src/models/weights/<model_id>/, e.g.
            'src/models/weights/qwen3vl-2b-instruct-q4km/Qwen3VL-2B-Instruct-Q4_K_M.gguf'.
            Every model gets its own subfolder so multi-file models (a language
            model plus an mmproj/clip projector, LoRA adapters, etc.) never mix
            files between models. Unused for API providers.
        mmproj_path: (llama_cpp only) path to the CLIP/vision projector .gguf
            ("mmproj") file, required by llama.cpp to actually run vision input.
            Only meaningful when 'vision' or 'omni' is in capabilities.
        context_length: native max context length the model was trained/tuned for.
        parameters: parameter count, e.g. '2B'. May be None/unknown for API models.
        license: model license, e.g. 'Apache-2.0'.
        source: where the weights/model came from, e.g. a Hugging Face repo id
            or the API provider's own model catalog page.
        description: free-text model card notes (strengths, intended use, etc.).
        load_params: (llama_cpp only) overrides merged over LOAD_PARAM_DEFAULTS
            for Llama.__init__.
        sampling_params: overrides merged over SAMPLING_PARAM_DEFAULTS — used by
            both providers (temperature, max_tokens, etc. are common concepts;
            an adapter ignores any keys it can't map onto its own API).
        provider_config: (openai_compatible only) dict of
            {"base_url": ..., "api_key_env": "OPENAI_API_KEY", "model_name": "gpt-4o-mini"}.
            api_key_env names an environment variable read at load time — the
            key itself is never stored in the model card or written to disk.
        """
        unknown_caps = set(capabilities) - KNOWN_CAPABILITIES
        if unknown_caps:
            raise ValueError(f"Unknown capabilities {unknown_caps} for model '{model_id}'. Known: {KNOWN_CAPABILITIES}")
        if provider not in KNOWN_PROVIDERS:
            raise ValueError(f"Unknown provider '{provider}' for model '{model_id}'. Known: {KNOWN_PROVIDERS}")
        if provider == "llama_cpp" and not path:
            raise ValueError(f"Model '{model_id}' uses provider 'llama_cpp' but no 'path' was given.")
        if provider == "openai_compatible":
            cfg = provider_config or {}
            missing = [k for k in ("base_url", "api_key_env", "model_name") if k not in cfg]
            if missing:
                raise ValueError(f"Model '{model_id}' uses provider 'openai_compatible' but provider_config is missing {missing}.")

        self.models[model_id] = {
            "id": model_id,
            "name": name,
            "family": family,
            "quantization": quantization,
            "capabilities": list(capabilities),
            "provider": provider,
            "path": path,
            "mmproj_path": mmproj_path,
            "context_length": context_length,
            "parameters": parameters,
            "license": license,
            "source": source,
            "description": description,
            "load_params": {**LOAD_PARAM_DEFAULTS, **(load_params or {})},
            "sampling_params": {**SAMPLING_PARAM_DEFAULTS, **(sampling_params or {})},
            "provider_config": provider_config or {},
        }

    def get(self, model_id: str) -> dict:
        if model_id not in self.models:
            raise ValueError(f"Unknown model '{model_id}'. Available: {list(self.models.keys())}")
        return self.models[model_id]

    def list_ids(self) -> list[str]:
        return list(self.models.keys())


registry = ModelRegistry()
