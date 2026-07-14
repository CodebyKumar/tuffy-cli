"""Public package surface for external consumers (e.g. tuffy-ui/backend).

Everything under src/ is internal — this is the only supported import path
from outside the tuffy repo:

    from tuffy import create_session, AgentSession, run_turn_stream

No new dependencies are introduced here; this module only wraps existing
src.cli.session.Session / src.cli.turn / src.engine.turn_engine machinery
that main.py already assembles ad hoc for the terminal.
"""

from typing import Iterator, Optional

from src.engine.events import TurnEvent

__all__ = ["create_session", "AgentSession", "run_turn_stream"]


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

    def system_message(self, context_plan=None) -> dict:
        return self._session.system_message(context_plan=context_plan)

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
