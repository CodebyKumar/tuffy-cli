# src/prompts/

Every system-prompt string, centralized. `src/engine/` and `main.py` only ever call
`build_system_prompt()` or a `templates.*` function from this package — neither constructs or
holds a raw prompt string itself.

- [personas.yaml](personas.yaml) - static persona/system-prompt presets. Edit this to change
  tone or top-level rules; the active preset is picked by the `active:` key (or overridden by
  `build_system_prompt(preset_name=...)`).
- [templates.py](templates.py) - Python-built prompt fragments: `self_model()` (identity +
  capabilities, sourced from [src/identity.py](../identity.py)), `runtime_context()` (tool
  signatures, memory, session summaries, lessons, skills list), the generic ReAct examples, and
  the tool-output/error framing strings the turn engine appends mid-turn.
- [__init__.py](__init__.py) - `load_preset()` + `build_system_prompt()`, which stitches a
  persona preset, the self-model, and the runtime context into the final system prompt string.

## Why the ReAct examples are tool-agnostic

`templates.py`'s few-shot examples use placeholders (`<tool name from the list>`, `<value taken
from the user's actual message>`) instead of real tool names/arguments. An earlier version used
concrete examples and a small model parroted the example's literal values back as if they were
real input (it once called `view_image('/Users/me/Desktop/screenshot.png')` unprompted). Keep
new examples generic for the same reason.

## Adding a new prompt fragment

Write a function in `templates.py` that returns a string, and call it from `build_system_prompt()`
or from wherever in `src/engine/turn_engine.py` needs to inject it mid-conversation (e.g.
`tool_output_prompt`, `force_final_answer`). Never inline a prompt string directly in
`src/engine/`.
