"""Registers every model Tuffy can load. Model cards live in src/models/configs/
(local.py for llama.cpp/gguf weights, api.py for API providers) - add a new
model by calling registry.register(...) in the appropriate file; it becomes
available to /models automatically, no other wiring needed.
"""

import src.models.configs.local  # noqa: F401 - registers local gguf models as a side effect of import
import src.models.configs.api  # noqa: F401 - registers API-provider models as a side effect of import

DEFAULT_MODEL = "qwen3vl-2b-instruct-q4km"
