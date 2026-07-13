"""End-to-end tests for turn_engine.run_turn: the ReAct loop driven against a
scripted FakeProvider, asserting on the actual event stream a real caller
(src/cli/turn.py) would consume. Covers the normal path, tool-call hops,
tool failures/self-correction, the exact-repeat guard, the hop-budget
exhaustion path, and every failure mode identified in the pre-redesign
audit (OOM, provider error, degenerate reply, foreign script, an
arbitrary-exception tool that must NOT crash the generator)."""

import pytest

from src.engine import tool_dispatch, turn_engine
from src.engine.errors import OutOfMemoryError
from src.engine.events import Done, Failed, Status, Thought, Token, ToolCall, ToolResult
from src.llm.base import ProviderError
from src.tools.registry import registry
from tests.fakes import FakeProvider


@pytest.fixture
def clean_registry():
    saved_functions = dict(registry.functions)
    saved_schemas = list(registry.schemas)
    saved_groups = dict(registry.groups)
    yield registry
    registry.functions = saved_functions
    registry.schemas = saved_schemas
    registry.groups = saved_groups


def _register(name, func, required=None):
    registry.register(
        name=name,
        description="test tool",
        parameters={p: {"type": "string", "description": ""} for p in (required or [])},
        required=required or [],
        group="test",
    )(func)


def run(script, messages=None):
    provider = FakeProvider(script)
    messages = messages if messages is not None else [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]
    events = list(turn_engine.run_turn(provider, {}, messages))
    return events, messages


def only(events, cls):
    return [e for e in events if isinstance(e, cls)]


class TestNormalAnswer:
    def test_plain_answer_streams_tokens_then_done(self):
        events, messages = run(["<think>just chat</think>Hi there, how can I help?"])
        tokens = only(events, Token)
        assert "".join(t.text for t in tokens) == "Hi there, how can I help?"
        done = only(events, Done)
        assert len(done) == 1
        assert done[0].full_text == "Hi there, how can I help?"
        assert not only(events, Failed)

    def test_thought_is_traced_not_shown_as_answer(self):
        events, _ = run(["<think>internal reasoning</think>Visible answer"])
        thoughts = only(events, Thought)
        assert thoughts[0].text == "internal reasoning"
        tokens = only(events, Token)
        assert "internal reasoning" not in "".join(t.text for t in tokens)

    def test_done_event_carries_final_text_for_caller_to_persist(self):
        # run_turn deliberately does NOT append the final answer to
        # `messages` itself - same contract as the old run_stream: the
        # caller (src/cli/turn.py) owns writing the assistant turn to
        # history, since it also decides whether to roll the turn back
        # (e.g. an empty-response guard) before anything is persisted.
        events, messages = run(["<think>ok</think>Final text"])
        done = only(events, Done)
        assert done[0].full_text == "Final text"
        assert not any(m.get("role") == "assistant" for m in messages)


class TestToolCallFlow:
    def test_successful_tool_call_then_final_answer(self, clean_registry):
        _register("get_time", lambda: "3:00pm", required=[])
        script = [
            '<think>need the time</think><tool_call>{"thought": "check time", "name": "get_time", "arguments": {}}</tool_call>',
            "<think>got it</think>It's 3:00pm.",
        ]
        events, messages = run(script)
        calls = only(events, ToolCall)
        results = only(events, ToolResult)
        assert calls[0].name == "get_time"
        assert results[0].ok
        assert results[0].result == "3:00pm"
        done = only(events, Done)
        assert done[0].full_text == "It's 3:00pm."

    def test_tool_call_thought_is_carried_on_the_event_not_the_spinner(self, clean_registry):
        """Regression test: the <tool_call> JSON's own "thought" field used
        to get fed into a Status event (spinner label only) - it would flash
        on screen for a fraction of a second as spinner text and vanish the
        instant the next event stopped the spinner, never appearing as a
        permanent [thought] line. It must now travel on ToolCall.thought so
        the renderer can print it permanently, same as a <think> block."""
        _register("get_time", lambda: "3:00pm", required=[])
        script = [
            '<tool_call>{"thought": "need the current time", "name": "get_time", "arguments": {}}</tool_call>',
            "It's 3:00pm.",
        ]
        events, messages = run(script)
        calls = only(events, ToolCall)
        assert calls[0].thought == "need the current time"
        statuses = only(events, Status)
        assert not any(s.label == "need the current time" for s in statuses), (
            "the tool call's thought must not be used as a transient spinner label"
        )

    def test_tool_output_fed_back_as_observation(self, clean_registry):
        _register("get_time", lambda: "3:00pm", required=[])
        script = [
            '<tool_call>{"name": "get_time", "arguments": {}}</tool_call>',
            "It's 3:00pm.",
        ]
        events, messages = run(script)
        observation_messages = [m for m in messages if m.get("role") == "user" and "3:00pm" in m.get("content", "")]
        assert observation_messages, "tool output should be fed back as a user-role Observation"


class TestToolFailureRecovery:
    def test_tool_exception_does_not_crash_the_generator(self, clean_registry):
        """The core regression test at the engine level: a tool raising an
        arbitrary exception must show up as a failed ToolResult and let the
        loop continue (or gracefully end), never propagate out of run_turn
        as a raw exception."""
        def flaky():
            raise ValueError("boom")
        _register("flaky", flaky, required=[])

        script = [
            '<tool_call>{"name": "flaky", "arguments": {}}</tool_call>',
            "<think>ok</think>Sorry, that didn't work.",
        ]
        events, messages = run(script)  # must not raise
        results = only(events, ToolResult)
        assert not results[0].ok
        assert "boom" in results[0].result
        done = only(events, Done)
        assert done[0].full_text == "Sorry, that didn't work."

    def test_unknown_tool_name_reported_and_recoverable(self, clean_registry):
        script = [
            '<tool_call>{"name": "does_not_exist", "arguments": {}}</tool_call>',
            "I don't have that capability.",
        ]
        events, messages = run(script)
        results = only(events, ToolResult)
        assert not results[0].ok
        assert "does not exist" in results[0].result

    def test_self_correction_after_failure_still_succeeds(self, clean_registry):
        calls = {"n": 0}
        def sometimes_fails():
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("first attempt fails")
            return "worked on retry"

        # First call fails validation (missing arg) is a different path;
        # here we simulate the model retrying the SAME tool with same args
        # after a real exception - call_signature would block an identical
        # repeat, so the second script line uses a tool that succeeds
        # instead, mirroring "self-correction via a different/fixed call".
        _register("flaky_once", sometimes_fails, required=[])
        script = [
            '<tool_call>{"name": "flaky_once", "arguments": {}}</tool_call>',
            '<tool_call>{"name": "flaky_once", "arguments": {"retry": "1"}}</tool_call>',
            "Done.",
        ]
        events, messages = run(script)
        results = only(events, ToolResult)
        assert results[0].ok is False
        assert results[1].ok is True
        assert results[1].result == "worked on retry"


class TestRepeatedCallGuard:
    def test_exact_repeat_is_blocked_without_executing(self, clean_registry):
        calls = {"n": 0}
        def counted():
            calls["n"] += 1
            return "result"
        _register("counted", counted, required=[])

        script = [
            '<tool_call>{"name": "counted", "arguments": {}}</tool_call>',
            '<tool_call>{"name": "counted", "arguments": {}}</tool_call>',
            "Final answer.",
        ]
        events, messages = run(script)
        results = only(events, ToolResult)
        # Only ONE ToolResult event - the second identical call never executes.
        assert len(results) == 1
        assert calls["n"] == 1

    def test_same_tool_different_args_still_runs_but_gets_reminded(self, clean_registry):
        """Regression test: the exact-repeat guard only catches identical
        arguments - a model reflexively re-running the same tool with
        slightly different (but equally pointless) arguments used to sail
        right past it and burn a hop for no new information. The call must
        still run (a genuinely different second use of the same tool is
        legitimate), but the Observation must remind the model what it
        already got back from the first call."""
        calls = {"n": 0}
        def search(query):
            calls["n"] += 1
            return f"result for {query}"
        _register("search", search, required=["query"])

        script = [
            '<tool_call>{"name": "search", "arguments": {"query": "first powerbank ever made"}}</tool_call>',
            '<tool_call>{"name": "search", "arguments": {"query": "first power bank invented"}}</tool_call>',
            "Final answer.",
        ]
        events, messages = run(script)
        results = only(events, ToolResult)
        # BOTH calls actually ran (different arguments, not blocked)...
        assert len(results) == 2
        assert calls["n"] == 2
        # ...but the second call's Observation reminds the model it already
        # called this tool and got a result.
        reminders = [
            m for m in messages
            if m.get("role") == "user"
            and "already called search once this turn" in m.get("content", "")
        ]
        assert reminders, "expected a reminder prefix on the second call's Observation"
        assert "result for first powerbank ever made" in reminders[0]["content"], (
            "the reminder must restate the FIRST call's actual result"
        )


class TestDegenerateReply:
    def test_degenerate_reply_triggers_retry_not_shown_to_user(self):
        script = ["user", "<think>ok</think>Real answer now."]
        events, messages = run(script)
        tokens = only(events, Token)
        assert "".join(t.text for t in tokens) == "Real answer now."
        done = only(events, Done)
        assert done[0].full_text == "Real answer now."


class TestForeignScript:
    def test_unsourced_foreign_script_redirects_to_translate(self, clean_registry):
        script = [
            "Hello\nこんにちは friend",
            "<think>ok</think>Hello, friend!",
        ]
        events, messages = run(script)
        done = only(events, Done)
        assert done[0].full_text == "Hello, friend!"
        translate_prompts = [m for m in messages if "translate" in m.get("content", "")]
        assert translate_prompts


class TestHopBudget:
    def test_forced_final_answer_when_hops_exhausted(self, clean_registry):
        _register("noop", lambda: "did nothing useful", required=[])
        call = '<tool_call>{"name": "noop", "arguments": {"n": "%d"}}</tool_call>'
        # _MAX_TOOL_HOPS = 4: hops 0-3 are the tool calls this script drives
        # (distinct arguments each time to dodge the exact-repeat guard);
        # hop 3 is_last_hop, so after that tool succeeds the engine forces
        # one extra completion (a 5th script entry) for the final answer.
        script = [
            call % 1, call % 2, call % 3, call % 4,
            "forced final answer text",
        ]
        events, messages = run(script)
        done = only(events, Done)
        assert len(done) == 1
        assert done[0].full_text == "forced final answer text"

    def test_fallback_sentence_when_model_produces_nothing_at_hop_limit(self, clean_registry):
        _register("noop", lambda: "x", required=[])
        call = '<tool_call>{"name": "noop", "arguments": {"n": "%d"}}</tool_call>'
        script = [call % 1, call % 2, call % 3, call % 4, ""]
        events, messages = run(script)
        done = only(events, Done)
        assert "wasn't able to finish" in done[0].full_text


class TestProviderFailures:
    def test_bare_runtime_error_is_not_miscategorized_as_oom(self):
        # Regression test for the old bug: turn.py used to catch ANY bare
        # RuntimeError and report it as "machine is low on memory" - even
        # one raised by unrelated code. The new engine only recognizes the
        # specific OutOfMemoryError type that LlamaCppProvider raises after
        # positively identifying a decode failure; a bare RuntimeError from
        # anywhere else must propagate as itself, not be silently relabeled.
        with pytest.raises(RuntimeError):
            list(turn_engine.run_turn(FakeProvider([RuntimeError("unrelated bug")]), {}, [
                {"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            ]))

    def test_out_of_memory_error_from_provider_becomes_failed_event(self):
        events, _ = run([OutOfMemoryError("decode -3")])
        failed = only(events, Failed)
        assert len(failed) == 1
        assert failed[0].kind == "oom"

    def test_provider_error_becomes_failed_event(self):
        events, _ = run([ProviderError("rate limited")])
        failed = only(events, Failed)
        assert len(failed) == 1
        assert failed[0].kind == "provider"
        assert "rate limited" in failed[0].message

    def test_failure_ends_the_stream_no_done_event(self):
        events, _ = run([OutOfMemoryError("x")])
        assert not only(events, Done)
        assert len(only(events, Failed)) == 1
