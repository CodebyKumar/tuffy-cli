"""Single entry point for everything prompt-related.

- personas.yaml   - static persona/system-prompt presets (edit this to change tone/rules).
- templates.py    - Python-built prompt fragments (self-model, runtime context,
                    ReAct observation framing, error messages).

src/agent.py and main.py only ever call build_system_prompt() / the
templates.* functions from this package — they never construct or hold a raw
prompt string themselves.
"""

import os
import yaml

from src.memory import load_memory, load_sessions, load_lessons
from src.tools.registry import registry
from src.prompts import templates

_PERSONAS_YAML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "personas.yaml")


def load_preset(name: str = None) -> str:
    """Loads a named preset's system_prompt string from personas.yaml. Defaults to the file's 'active' preset."""
    with open(_PERSONAS_YAML_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    preset_name = name or data.get("active", "tuffy")
    presets = data.get("presets", {})
    if preset_name not in presets:
        raise ValueError(f"Persona preset '{preset_name}' not found in personas.yaml. Available: {list(presets.keys())}")
    return presets[preset_name]["system_prompt"]


def build_system_prompt(preset_name: str = None, model_card: dict = None) -> str:
    """Renders a named personas.yaml preset plus the agent's self-model and
    the current runtime context (tool signatures, memory, past sessions,
    lessons, protocol examples) into the full system prompt."""
    sections = [load_preset(preset_name)]
    if model_card:
        sections.append(templates.self_model(model_card))
    sections.append(templates.runtime_context(
        tool_lines=registry.tool_lines(),
        known_facts=load_memory(),
        session_summaries=load_sessions(),
        lessons=load_lessons(),
    ))
    return "\n\n".join(sections)
