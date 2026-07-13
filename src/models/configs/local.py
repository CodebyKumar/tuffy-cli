"""Registers every locally-loaded (llama.cpp/gguf) model. Add a new local
model by calling registry.register(...) here with its full model card - it
becomes available to /models automatically, no other wiring needed.
"""

from src.models.registry import registry

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
    context_length=8192,
    parameters="2B",
    license="Apache-2.0",
    source="Qwen/Qwen3-VL-2B-Instruct",
    description="Small instruction-tuned vision-language model; used as Tuffy's default local model.",
    load_params={
        # Doubled from 4096: Tuffy's system prompt (persona + tool catalog)
        # already costs ~2500-2600 tokens, leaving elastimem's memory
        # sections (facts/episodic/sessions/lessons combined) only ~400
        # tokens of budget at 4096 - too tight for meaningful recall on any
        # of those sections. 8192 leaves ~4500 tokens for memory instead.
        "n_ctx": 8192,
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
        # The chat template's eos_token (<|im_end|>) is what should stop
        # generation, but a small quantized model occasionally drifts past it
        # and starts emitting the literal text of the NEXT turn's role marker
        # ("<|im_start|>user\n...") as if it were still answering — the
        # observed symptom is a reply that's just the bare word "user". These
        # stop strings are a text-level backstop: the moment any of them
        # appears, llama.cpp cuts generation instead of letting it continue
        # into a leaked template fragment.
        "stop": ["<|im_start|>", "<|im_end|>"],
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
    context_length=8192,
    parameters="2B",
    license="Apache-2.0",
    source="Qwen/Qwen3-VL-2B-Instruct",
    description="Small instruction-tuned vision-language model; used as Tuffy's default local model.",
    load_params={
        # Doubled from 4096: Tuffy's system prompt (persona + tool catalog)
        # already costs ~2500-2600 tokens, leaving elastimem's memory
        # sections (facts/episodic/sessions/lessons combined) only ~400
        # tokens of budget at 4096 - too tight for meaningful recall on any
        # of those sections. 8192 leaves ~4500 tokens for memory instead.
        "n_ctx": 8192,
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
        # The chat template's eos_token (<|im_end|>) is what should stop
        # generation, but a small quantized model occasionally drifts past it
        # and starts emitting the literal text of the NEXT turn's role marker
        # ("<|im_start|>user\n...") as if it were still answering — the
        # observed symptom is a reply that's just the bare word "user". These
        # stop strings are a text-level backstop: the moment any of them
        # appears, llama.cpp cuts generation instead of letting it continue
        # into a leaked template fragment.
        "stop": ["<|im_start|>", "<|im_end|>"],
    },
)
