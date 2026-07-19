"""Public package surface for external consumers (e.g. tuffy-ui/backend).

Everything under src/ is internal — this is the only supported import path
from outside the tuffy-cli repo:

    from tuffy_core import create_session, AgentSession, run_turn_stream

No new dependencies are introduced here; this module only wraps existing
src.cli.session.Session / src.cli.turn / src.engine.turn_engine machinery
that main.py already assembles ad hoc for the terminal.
"""

from typing import Iterator, Optional

from src.engine.events import TurnEvent

__all__ = [
    "create_session",
    "AgentSession",
    "run_turn_stream",
    "list_tools",
    "memory_summary",
    "memory_search",
    "list_skills",
    "list_models",
    "WhisperSTT",
    "PiperTTS",
]


class AgentSession:
    """Thin public wrapper around src.cli.session.Session, exposing only
    what an external caller needs — not Session's internal fields
    (agent, pending_image_data_uri, captured_images, health)."""

    def __init__(self, session):
        self._session = session

    @property
    def history(self) -> list:
        """The live chat history list. Callers should treat this as
        read-only — run_turn_stream mutates the same underlying list as
        it streams, which is what lets tool-call scaffolding and the
        final answer land in history without a separate write-back step."""
        return self._session.messages

    def switch_model(self, model_id: str) -> None:
        self._session.switch_model(model_id)

    def run_command(self, text: str) -> dict:
        """Runs a slash command exactly as the terminal would
        (src/cli/commands.py's handle_command), capturing its stdout instead
        of printing to a real terminal - this is the single generic entry
        point a UI needs for the whole command set (/memory, /mcp, /models,
        /tools, /skills, /status, /purge, /new, /clear, /image, ...) rather
        than one bespoke REST endpoint per command. Returns
        {"output": str, "exit": bool} - "exit" is True for /exit or /quit,
        which the terminal handles by tearing down the process; a UI caller
        decides what that means for itself (e.g. ignore it, since a desktop
        app shouldn't quit just because the user typed /exit in a chat box).
        Raises ValueError if `text` isn't recognized as a slash command."""
        import contextlib
        import io
        import re

        from src.cli.commands import handle_command

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = handle_command(self._session, text.strip())
        if result == "unhandled":
            raise ValueError(f"Unknown command: {text}")
        # Strip ANSI color codes (src/cli/display.py's C_* constants) - they
        # style a real terminal, but a UI caller renders this text in HTML,
        # where raw escape sequences would show up as garbage characters.
        output = re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue())
        return {"output": output, "exit": result == "exit"}

    def system_message(self, context_plan=None) -> dict:
        return self._session.system_message(context_plan=context_plan)

    def attach_image(self, data_uri: str) -> None:
        """Stages an already-encoded image data URI to attach to the next
        user turn — same effect as the terminal's /image command
        (src/cli/commands.py's cmd_image), minus the file-path decode step
        since a UI caller already has the data URI (e.g. from a browser
        FileReader). Raises ValueError if the active model has no vision
        capability, matching cmd_image's own guard."""
        if not self._session.agent.supports_vision:
            raise ValueError(
                f"Model '{self._session.current_model_id}' has no vision capability."
            )
        self._session.pending_image_data_uri = data_uri

    @property
    def mcp_servers(self) -> list[dict]:
        """Connected MCP servers this session sees, as
        [{"name": str, "tool_count": int}, ...] - mirrors /mcp's listing."""
        from src.tools.registry import registry as tool_registry

        servers: dict[str, int] = {}
        for group, schemas in tool_registry.tools_by_group():
            if not group.startswith("mcp:"):
                continue
            servers[group[len("mcp:"):]] = len(schemas)
        return [{"name": name, "tool_count": count} for name, count in servers.items()]

    def status(self) -> dict:
        """Session status snapshot - mirrors /status's fields exactly
        (src/cli/commands.py's cmd_status), as a dict instead of print()s."""
        from src.cli.session import estimate_tokens
        from src.models.registry import registry as model_registry

        session = self._session
        card = model_registry.get(session.current_model_id)
        turns = sum(1 for m in session.messages if m["role"] == "user")
        used_tokens = estimate_tokens(session.messages)
        context_length = card.get("context_length")
        return {
            "active_model_id": session.current_model_id,
            "model_name": card["name"],
            "provider": card["provider"],
            "vision_enabled": session.agent.supports_vision,
            "turn_count": turns,
            "pending_image": bool(session.pending_image_data_uri),
            "recent_outcomes": session.health.recent_outcomes(),
            "used_tokens": used_tokens,
            "context_length": context_length,
            "context_usage_pct": (
                round(used_tokens / context_length * 100, 1) if context_length else None
            ),
        }

    def end(self) -> None:
        """Frees the loaded model and writes this session into episodic
        memory. Callers embedding tuffy in a longer-lived process (e.g. a
        backend server) must call this before process exit — see main.py's
        own shutdown path for why (llama.cpp's Metal static destructors)."""
        self._session.end()


def create_session(model_id: Optional[str] = None) -> AgentSession:
    """Builds a ready-to-use session: loads the given model (or the
    persisted/default model if none given), attaches it to Elastimem, and
    reconfigures Elastimem's token budget for it — the same sequence
    main.py performs inline at startup and Session.switch_model repeats on
    every model change.

    Unlike main.py's startup path, this does not silently fall back to a
    different model on failure — that's terminal UX policy, not core
    session construction. Callers here get a raised exception and decide
    their own fallback behavior, if any."""
    from src.models import DEFAULT_MODEL
    from src.settings import get_default_model
    from src.cli.session import Session
    from src.models.registry import registry as model_registry
    from src.prompts import build_system_prompt
    import src.memory as memory

    resolved_model_id = model_id or get_default_model() or DEFAULT_MODEL
    session = Session(resolved_model_id)

    memory.attach_llm(session.agent.complete)
    model_card = model_registry.get(session.current_model_id)
    static_tokens = len(build_system_prompt(model_card=model_card)) // 4
    memory.reconfigure_for_model(model_card, static_prompt_tokens=static_tokens)

    return AgentSession(session)


def run_turn_stream(session: AgentSession, text: str) -> Iterator[TurnEvent]:
    """Runs one user turn against `session`, yielding each TurnEvent as it
    happens. Same history-trimming/health-tracking/memory-recording/
    rollback-on-failure behavior as the terminal's run_turn — this calls
    the exact same shared generator, just without stdout rendering."""
    from src.cli.turn import _run_turn_events

    yield from _run_turn_events(session._session, text)


def list_tools() -> list[dict]:
    """All registered tools grouped by category, as a flat list of
    {"name", "group", "description"} - mirrors /tools' CLI output
    (src/cli/commands.py's cmd_tools) without the print()s."""
    from src.tools.registry import registry as tool_registry

    tools = []
    for group, schemas in tool_registry.tools_by_group():
        for schema in schemas:
            fn = schema["function"]
            tools.append({"name": fn["name"], "group": group, "description": fn["description"]})
    return tools


def memory_summary() -> dict:
    """Long-term memory status - mirrors /memory's no-argument output
    (src/cli/commands.py's cmd_memory) as a dict."""
    import src.memory as memory

    stats = memory.mem.stats()
    profile = memory.mem.profile
    return {
        "tier": profile.tier.name,
        "fact_count": stats.get("facts", 0),
        "session_count": stats.get("sessions", 0),
        "lesson_count": stats.get("lessons", 0),
        "db_path": stats.get("path"),
        "db_size_bytes": stats.get("db_bytes", 0),
    }


def memory_search(query: str) -> list[dict]:
    """Semantic search over past conversations + facts - mirrors
    /memory search <query>'s output as a list of
    {"text", "kind", "date", "score"}."""
    import src.memory as memory

    hits = memory.mem.recall(query)
    return [
        {"text": hit.text, "kind": hit.kind, "date": hit.date, "score": hit.score}
        for hit in hits
    ]


def list_skills() -> list[dict]:
    """Installed skills - mirrors /skills' output
    (src/cli/commands.py's cmd_skills) as a list of {"name", "description"}."""
    from src.skills.loader import list_skills as _list_skills

    return [
        {"name": name, "description": info["description"]}
        for name, info in _list_skills().items()
    ]


def list_models() -> list[dict]:
    """All available model cards (local + API) - mirrors /models' listing
    (src/cli/commands.py's cmd_models, no-argument branch)."""
    from src.models.registry import registry as model_registry

    return [model_registry.get(mid) for mid in model_registry.list_ids()]


class WhisperSTT:
    """Wrapper for Whisper Speech-to-Text, loaded lazily to avoid import errors
    if voice dependencies are not installed."""

    def __init__(self, model_name: str = "small.en"):
        from src.voice.stt import WhisperSTT as _WhisperSTT
        self._impl = _WhisperSTT(model_name)

    def transcribe(self, pcm) -> str:
        return self._impl.transcribe(pcm)


class PiperTTS:
    """Wrapper for Piper Text-to-Speech, loaded lazily to avoid import errors
    if voice dependencies are not installed."""

    def __init__(self, voice_id: str = "en_US-lessac-medium"):
        from src.voice.tts import PiperTTS as _PiperTTS
        self._impl = _PiperTTS(voice_id)
        self.sample_rate = self._impl.sample_rate
        self.sample_width = self._impl.sample_width
        self.channels = self._impl.channels

    def synthesize(self, text: str):
        return self._impl.synthesize(text)
