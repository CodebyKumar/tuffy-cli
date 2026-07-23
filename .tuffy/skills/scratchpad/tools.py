"""Scratchpad skill tools: a tiny JSON-backed note list, scoped to this
skill's own folder so it survives restarts without touching long-term
memory (Elastimem) or the chat history."""

import json
import os
from datetime import datetime, timezone

from src.tools.registry import registry

_NOTES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notes.json")


def _load_notes() -> list[dict]:
    if not os.path.isfile(_NOTES_PATH):
        return []
    with open(_NOTES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_notes(notes: list[dict]) -> None:
    with open(_NOTES_PATH, "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2)


@registry.register(
    name="scratch_note",
    description="Add a short-lived scratch note (e.g. a TODO or reminder for this project). Not long-term memory — use 'remember' for durable facts about the user.",
    parameters={"text": {"type": "string", "description": "The note text to record."}},
    required=["text"],
    group="docs",
)
def scratch_note(text: str) -> str:
    notes = _load_notes()
    notes.append({"text": text, "ts": datetime.now(timezone.utc).isoformat()})
    _save_notes(notes)
    return f"Noted ({len(notes)} note(s) total)."


@registry.register(
    name="scratch_list",
    description="List all scratch notes recorded so far, oldest first.",
    parameters={},
    required=[],
    group="docs",
)
def scratch_list() -> str:
    notes = _load_notes()
    if not notes:
        return "No scratch notes yet."
    return "\n".join(f"[{n['ts']}] {n['text']}" for n in notes)


@registry.register(
    name="scratch_clear",
    description="Clear all scratch notes. Destructive — confirm with the user first unless they explicitly asked to clear/reset the scratchpad.",
    parameters={},
    required=[],
    group="docs",
)
def scratch_clear() -> str:
    count = len(_load_notes())
    _save_notes([])
    return f"Cleared {count} note(s)."
