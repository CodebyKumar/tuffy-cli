"""Integration smoke test for the full user-input-to-response pipeline:
real Session, real src.cli.turn.run_turn, real Spinner/display code, real
elastimem memory wiring — everything except the actual model weights, which
this sandboxed test environment has neither local gguf files nor an API key
for. A FakeProvider stands in for LlamaCppProvider/OpenAICompatibleProvider
underneath a real ModelAgent, so this exercises the exact code path
main.py's input loop drives, just with scripted model output instead of a
real completion.

This is the closest thing to "run the real app and talk to it" achievable
without model weights or network access - it proves the new pipeline is
wired together correctly end to end, not just that its pieces work in
isolation (which the other test files already cover)."""

import io
import contextlib

import pytest

from src.cli.session import Session, TurnHealth
from src.cli.turn import run_turn
from src.engine.model_agent import ModelAgent
from src.models.registry import registry as model_registry
from src.tools.registry import registry
from tests.fakes import FakeProvider

_FAKE_MODEL_ID = "fake-test-model"


@pytest.fixture(autouse=True)
def isolated_memory_db(tmp_path, monkeypatch):
    """run_turn calls memory.mem.record_turn()/build_context() on the
    module-level singleton, which points at the REAL ./data/memory/tuffy.db.
    Without this fixture, every test turn here gets persisted into the
    user's actual memory database — junk turns like 'do the broken thing'
    then surface in real episodic retrieval and degrade real answers
    (observed in practice). Swap in a throwaway store for the duration of
    each test."""
    import elastimem
    import src.memory as memory
    isolated = elastimem.open(str(tmp_path / "test-mem.db"), context_tokens=4096)
    monkeypatch.setattr(memory, "mem", isolated)
    yield
    isolated.close()


@pytest.fixture
def clean_registry():
    saved_functions = dict(registry.functions)
    saved_schemas = list(registry.schemas)
    saved_groups = dict(registry.groups)
    yield registry
    registry.functions = saved_functions
    registry.schemas = saved_schemas
    registry.groups = saved_groups


@pytest.fixture
def fake_model_card():
    """run_turn rebuilds the real system prompt every turn (see
    session.system_message), which needs a real entry in the model
    registry - register a minimal local-style card for the test model id
    and clean it up afterward so it doesn't leak into other tests or show
    up in a real /models listing."""
    model_registry.register(
        model_id=_FAKE_MODEL_ID,
        name="Fake Test Model",
        family="test",
        quantization="none",
        capabilities=["text"],
        # "openai_compatible" needs provider_config but not a real weights
        # file on disk (llama_cpp requires 'path' to point at one) - this
        # card is never actually loaded through build_provider(), only read
        # for its metadata (build_system_prompt, /status token estimates).
        provider="openai_compatible",
        provider_config={"base_url": "http://unused", "api_key_env": "UNUSED", "model_name": "unused"},
        context_length=4096,
        sampling_params={},
    )
    yield
    model_registry.models.pop(_FAKE_MODEL_ID, None)


def _fake_session(script) -> Session:
    """Builds a real Session without going through load_agent (which would
    try to load a real model) - constructs ModelAgent's __new__ directly and
    swaps in a FakeProvider, matching how the app wires session.agent.provider
    everywhere else."""
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
    return session


def _run_silently(session, user_input):
    """run_turn writes directly to stdout (spinner + live tokens) - capture
    it so pytest output stays clean, but keep the buffer to assert on if a
    test wants to inspect what was actually rendered."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ok = run_turn(session, user_input)
    return ok, buf.getvalue()


class TestFullPipelinePlainAnswer:
    def test_user_input_to_rendered_response(self, monkeypatch, fake_model_card):
        monkeypatch.setenv("TUFFY_NO_AUTO_MEMORY", "1")
        session = _fake_session(["<think>simple greeting</think>Hello! How can I help you today?"])
        ok, output = _run_silently(session, "hi there")

        assert ok is True
        assert "Hello! How can I help you today?" in output
        # The user's message and the model's answer both landed in history.
        assert session.messages[-2] == {"role": "user", "content": "hi there"}
        assert session.messages[-1] == {"role": "assistant", "content": "Hello! How can I help you today?"}

    def test_thought_not_leaked_into_saved_history(self, monkeypatch, fake_model_card):
        monkeypatch.setenv("TUFFY_NO_AUTO_MEMORY", "1")
        session = _fake_session(["<think>internal reasoning nobody should see</think>Visible reply"])
        ok, output = _run_silently(session, "question")
        assert ok is True
        assert session.messages[-1]["content"] == "Visible reply"
        assert "internal reasoning" not in session.messages[-1]["content"]


class TestFullPipelineToolCall:
    def test_tool_call_round_trip(self, monkeypatch, clean_registry, fake_model_card):
        monkeypatch.setenv("TUFFY_NO_AUTO_MEMORY", "1")
        registry.register(
            name="get_joke",
            description="test tool",
            parameters={},
            required=[],
            group="test",
        )(lambda: "why did the chicken cross the road? to get to the other side.")

        session = _fake_session([
            '<think>they want a joke</think><tool_call>{"thought": "fetch a joke", "name": "get_joke", "arguments": {}}</tool_call>',
            "<think>relay it</think>Here's one: why did the chicken cross the road? To get to the other side!",
        ])
        ok, output = _run_silently(session, "tell me a joke")

        assert ok is True
        assert "[execute] get_joke" in output
        assert "[result]" in output
        assert "why did the chicken cross the road" in output
        # Compacted history: intermediates dropped, only Q + final A remain
        # (plus the system message at index 0).
        assert session.messages[-2] == {"role": "user", "content": "tell me a joke"}
        assert "chicken" in session.messages[-1]["content"]


class TestFullPipelineToolFailureDoesNotCrash:
    def test_buggy_tool_recovers_gracefully(self, monkeypatch, clean_registry, fake_model_card):
        """The end-to-end version of the highest-severity regression test:
        drive an actual turn through the real CLI entry point where a tool
        raises an arbitrary exception, and confirm the whole pipeline
        recovers with a normal answer instead of an unhandled crash."""
        monkeypatch.setenv("TUFFY_NO_AUTO_MEMORY", "1")

        def broken_tool():
            raise KeyError("some internal bug having nothing to do with the model")

        registry.register(
            name="broken_tool",
            description="test tool that always fails",
            parameters={},
            required=[],
            group="test",
        )(broken_tool)

        session = _fake_session([
            '<tool_call>{"name": "broken_tool", "arguments": {}}</tool_call>',
            "<think>that failed, apologize</think>Sorry, I couldn't complete that.",
        ])
        ok, output = _run_silently(session, "do the broken thing")

        assert ok is True  # turn still completes successfully
        assert "Sorry, I couldn't complete that." in output
        assert session.messages[-1]["content"] == "Sorry, I couldn't complete that."


class TestFullPipelineFailure:
    def test_provider_failure_rolls_back_turn(self, monkeypatch, fake_model_card):
        monkeypatch.setenv("TUFFY_NO_AUTO_MEMORY", "1")
        from src.llm.base import ProviderError

        session = _fake_session([ProviderError("rate limited")])
        conversation_before = list(session.messages[1:])  # everything but the system message
        ok, output = _run_silently(session, "hello")

        assert ok is False
        assert "Generation failed" in output
        # Turn rolled back - no dangling user message with no answer. (The
        # system message itself is legitimately rebuilt every turn with a
        # fresh memory context plan, even on a rolled-back turn - same as
        # the old implementation - so only the conversational tail matters
        # here.)
        assert session.messages[1:] == conversation_before
