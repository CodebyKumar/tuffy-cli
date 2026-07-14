"""Tests for the tuffy/__init__.py public package surface — the exact
import path an external consumer (tuffy-ui/backend) uses:

    from tuffy import create_session, AgentSession, run_turn_stream

run_turn_stream is expected to behave identically to src.cli.turn.run_turn
(same history mutation, health tracking, rollback-on-failure) since both
now share src.cli.turn._run_turn_events — this file exercises that
contract directly against AgentSession rather than a raw Session, using
the same FakeProvider/__new__-bypass pattern as
tests/test_cli_turn_integration.py so no real model weights load."""

import pytest

from src.cli.session import Session, TurnHealth
from src.engine.events import Done, Failed, Token
from src.engine.model_agent import ModelAgent
from src.models.registry import registry as model_registry
from tests.fakes import FakeProvider
from tuffy import AgentSession, create_session, run_turn_stream

_FAKE_MODEL_ID = "fake-public-api-model"


@pytest.fixture(autouse=True)
def isolated_memory_db(tmp_path, monkeypatch):
    """Same rationale as test_cli_turn_integration.py's fixture of the same
    name: run_turn_stream drives memory.mem.record_turn()/build_context()
    on the real module-level singleton unless it's swapped out."""
    import elastimem
    import src.memory as memory
    isolated = elastimem.open(str(tmp_path / "test-mem.db"), context_tokens=4096)
    monkeypatch.setattr(memory, "mem", isolated)
    yield
    isolated.close()


@pytest.fixture
def fake_model_card():
    model_registry.register(
        model_id=_FAKE_MODEL_ID,
        name="Fake Public API Model",
        family="test",
        quantization="none",
        capabilities=["text"],
        provider="openai_compatible",
        provider_config={"base_url": "http://unused", "api_key_env": "UNUSED", "model_name": "unused"},
        context_length=4096,
        sampling_params={},
    )
    yield
    model_registry.models.pop(_FAKE_MODEL_ID, None)


def _fake_agent_session(script) -> AgentSession:
    """Mirrors test_cli_turn_integration.py's _fake_session, wrapped in the
    public AgentSession — the point of this helper is to get an AgentSession
    without going through create_session()'s real Session(model_id) call,
    which would try to load actual model weights."""
    session = Session.__new__(Session)
    agent = ModelAgent.__new__(ModelAgent)
    agent.model_id = _FAKE_MODEL_ID
    agent.sampling_params = {}
    agent.provider = FakeProvider(script)
    session.agent = agent
    session.current_model_id = _FAKE_MODEL_ID
    session.pending_image_data_uri = None
    session.captured_images = []
    session.health = TurnHealth()
    session.messages = [session.system_message()]
    return AgentSession(session)


class TestPublicSurfaceImportable:
    def test_expected_names_are_importable(self):
        import tuffy
        assert hasattr(tuffy, "create_session")
        assert hasattr(tuffy, "AgentSession")
        assert hasattr(tuffy, "run_turn_stream")

    def test_agent_session_does_not_expose_internal_fields(self, fake_model_card):
        session = _fake_agent_session(["hi"])
        assert not hasattr(session, "agent")
        assert not hasattr(session, "pending_image_data_uri")
        assert not hasattr(session, "captured_images")
        assert not hasattr(session, "health")


class TestRunTurnStream:
    def test_yields_token_and_done_events(self, monkeypatch, fake_model_card):
        monkeypatch.setenv("TUFFY_NO_AUTO_MEMORY", "1")
        session = _fake_agent_session(["<think>ok</think>Hello there!"])

        events = list(run_turn_stream(session, "hi"))

        assert any(isinstance(e, Token) for e in events)
        assert any(isinstance(e, Done) for e in events)
        # Same history-mutation contract as src.cli.turn.run_turn.
        assert session.history[-2] == {"role": "user", "content": "hi"}
        assert session.history[-1] == {"role": "assistant", "content": "Hello there!"}

    def test_failure_rolls_back_history(self, monkeypatch, fake_model_card):
        monkeypatch.setenv("TUFFY_NO_AUTO_MEMORY", "1")
        from src.llm.base import ProviderError

        session = _fake_agent_session([ProviderError("rate limited")])
        conversation_before = list(session.history[1:])

        events = list(run_turn_stream(session, "hello"))

        assert any(isinstance(e, Failed) for e in events)
        assert session.history[1:] == conversation_before

    def test_history_is_the_live_session_list(self, monkeypatch, fake_model_card):
        """AgentSession.history must be the same list run_turn_stream mutates
        in place, not a copy — external callers (e.g. a WS bridge) read this
        list directly to render/persist the transcript as it grows."""
        monkeypatch.setenv("TUFFY_NO_AUTO_MEMORY", "1")
        session = _fake_agent_session(["<think>ok</think>Hi!"])
        history_ref = session.history

        list(run_turn_stream(session, "hi"))

        assert history_ref is session.history
        assert history_ref[-1] == {"role": "assistant", "content": "Hi!"}


class TestCreateSession:
    def test_raises_on_unknown_model_instead_of_falling_back(self):
        """create_session must not replicate main.py's fallback-to-a-
        different-model UX policy — callers get a raised exception and
        decide their own fallback behavior."""
        with pytest.raises(Exception):
            create_session(model_id="definitely-not-a-registered-model-id")
