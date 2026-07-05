# src/models/

The model registry: every local gguf model Tuffy can load, switched between at runtime with `/models <id>` in the CLI. Mirrors how [src/tools/](../tools/) registers tools.

- [registry.py](registry.py) - `ModelRegistry` class, `register()` bookkeeping, and the full default parameter tables (`LOAD_PARAM_DEFAULTS`, `SAMPLING_PARAM_DEFAULTS`) covering every keyword argument `llama_cpp.Llama.__init__` and `create_chat_completion` accept.
- [__init__.py](__init__.py) - where models actually get registered via `registry.register(...)`. This is the only file you edit to add a model.
- `weights/<model_id>/` - one subfolder per registered model, holding that model's own file(s): the language-model `.gguf`, and (for vision models) its `mmproj-*.gguf` CLIP/vision projector. Never share a subfolder between two models — multi-file models (LM + mmproj, LM + LoRA, etc.) need every file scoped to that one model so switching models can't accidentally mix files.

## Adding a new model

1. Create `src/models/weights/<model_id>/` (pick `model_id` as `family-params-variant-quant`, all lowercase, e.g. `qwen3vl-4b-instruct-q4km` — fully spelled out so two quantizations of the same base model never collide or read as the same thing).
2. Download the model's `.gguf` weights file into that folder.
3. If the model is vision/audio/omni-capable, also download its matching `mmproj-*.gguf` (CLIP/audio projector) file into the same folder. This is a **separate file from the language model** — llama.cpp cannot do vision without it. It normally lives in the same Hugging Face repo as the language model, e.g. for Qwen3-VL-2B-Instruct: [Qwen/Qwen3-VL-2B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF) (see the `mmproj-*.gguf` asset in that repo).
4. In [__init__.py](__init__.py), call `registry.register(...)` with:
   - `model_id`, `name`, `family`, `quantization`, `path` (pointing at the file from step 2)
   - `capabilities` — include `"vision"` (or `"audio"`/`"omni"` as applicable) if you added an mmproj file
   - `mmproj_path` (pointing at the file from step 3), if applicable
   - the rest of the model card: `context_length`, `parameters`, `license`, `source`, `description`
   - `load_params`/`sampling_params` overrides for anything that shouldn't use the library default (see `LOAD_PARAM_DEFAULTS`/`SAMPLING_PARAM_DEFAULTS` in registry.py for the full list of tunables, e.g. `n_ctx`, `n_gpu_layers`, `temperature`, `repeat_penalty`)
5. Run the app and check `/models` lists it, then `/models <id>` to switch and load it.

No other file needs to change — `/models` in [main.py](../../main.py) reads the registry directly.
