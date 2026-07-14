"""tool_dispatch.execute: the boundary that must convert EVERY exception a
tool function can raise into a ToolExecutionError, never let one propagate
raw. This is the fix for the highest-severity bug found in the old
implementation (agent.py's _execute_tool_call only caught TypeError; any
other exception - ValueError, KeyError, a network error from an MCP tool -
crashed the whole session, not just the turn)."""

import pytest

from src.engine import tool_dispatch
from src.engine.errors import ToolExecutionError
from src.tools.registry import registry


@pytest.fixture
def clean_registry():
    """Tools register themselves at import time into the module-level
    singleton; save/restore around each test so registering a fake tool
    here doesn't leak into other tests or the real app."""
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


class TestParseToolCall:
    def test_valid_json(self):
        name, args, thought = tool_dispatch.parse_tool_call(
            '{"name": "foo", "arguments": {"x": 1}, "thought": "doing foo"}'
        )
        assert name == "foo"
        assert args == {"x": 1}
        assert thought == "doing foo"

    def test_malformed_json_raises_tool_execution_error(self):
        with pytest.raises(ToolExecutionError):
            tool_dispatch.parse_tool_call("{not json")

    def test_missing_name_raises(self):
        with pytest.raises(ToolExecutionError):
            tool_dispatch.parse_tool_call('{"arguments": {}}')

    def test_non_blank_name_parses_even_if_not_a_real_tool(self):
        """parse_tool_call only validates JSON shape - it has no hardcoded
        list of "fake" tool names (a small model echoing wording like "no
        tool needed" or a placeholder like "tool_name" into the name field
        both look the same to this function: a non-blank string). Whether
        the name is a REAL tool is a registry lookup, done by the caller
        (turn_engine), not a string blocklist here."""
        name, _, _ = tool_dispatch.parse_tool_call('{"name": "tool_name", "arguments": {}}')
        assert name == "tool_name"
        name, _, _ = tool_dispatch.parse_tool_call('{"name": "no tool needed", "arguments": {}}')
        assert name == "no tool needed"

    def test_empty_payload_gives_actionable_error(self):
        """Regression test: a truncated <tool_call> (closing tag never
        generated) used to surface as 'not valid JSON (Expecting value:
        line 1 column 1 (char 0))' — meaningless to the model reading it as
        an Observation. The error must restate the expected format."""
        with pytest.raises(ToolExecutionError) as exc_info:
            tool_dispatch.parse_tool_call("")
        assert "expected exactly" in str(exc_info.value)

    def test_json_surrounded_by_chatter_still_parses(self):
        """Small models wrap the JSON in prefix/suffix text — as long as one
        parseable object is in there, the call must run."""
        name, args, thought = tool_dispatch.parse_tool_call(
            'Sure, let me search:\n{"name": "web_search", "arguments": {"query": "mumbai"}}\nrunning now'
        )
        assert name == "web_search"
        assert args == {"query": "mumbai"}

    def test_truncated_tail_after_complete_object_still_parses(self):
        # e.g. the stream-parser fallback hands over "…object…\n</tool_ca"
        name, args, _ = tool_dispatch.parse_tool_call(
            '{"name": "web_search", "arguments": {"query": "mumbai"}}\n</tool_ca'
        )
        assert name == "web_search"

    def test_non_dict_arguments_rejected_with_format_hint(self):
        with pytest.raises(ToolExecutionError) as exc_info:
            tool_dispatch.parse_tool_call('{"name": "web_search", "arguments": "mumbai"}')
        assert "expected exactly" in str(exc_info.value)


class TestExecute:
    def test_unknown_tool_raises_tool_execution_error(self, clean_registry):
        with pytest.raises(ToolExecutionError):
            tool_dispatch.execute("definitely_not_a_real_tool", {}, "")

    def test_missing_required_argument_raises(self, clean_registry):
        _register("needs_arg", lambda x: x, required=["x"])
        with pytest.raises(ToolExecutionError):
            tool_dispatch.execute("needs_arg", {}, "")

    def test_successful_call_returns_events_and_output(self, clean_registry):
        _register("echo", lambda x: f"got {x}", required=["x"])
        events, output = tool_dispatch.execute("echo", {"x": "hi"}, "thinking")
        assert output == "got hi"
        assert any(type(e).__name__ == "ToolCall" for e in events)
        assert any(type(e).__name__ == "ToolResult" for e in events)

    def test_type_error_from_bad_call_is_wrapped(self, clean_registry):
        def strict(x, y):
            return "ok"
        _register("strict", strict, required=["x", "y"])
        # sanitization only filters to declared params, but a value that's
        # the wrong type and mishandled inside the function still surfaces
        # as a TypeError from *inside* the call in some cases; more directly
        # testable: pass through the signature-mismatch path.
        with pytest.raises(ToolExecutionError):
            tool_dispatch.execute("strict", {"x": "a"}, "")  # missing y -> ToolExecutionError before call

    def test_arbitrary_exception_from_tool_is_wrapped_not_propagated(self, clean_registry):
        """The core regression test: a tool raising ANY exception type must
        come back as ToolExecutionError, never escape as the raw exception."""
        def flaky(x):
            raise ValueError("boom, unrelated to argument validation")
        _register("flaky", flaky, required=["x"])

        with pytest.raises(ToolExecutionError) as exc_info:
            tool_dispatch.execute("flaky", {"x": "anything"}, "")
        assert "boom" in str(exc_info.value)
        assert exc_info.value.tool_name == "flaky"

    def test_key_error_from_tool_is_wrapped(self, clean_registry):
        def bad_lookup(x):
            return {}["missing_key"]
        _register("bad_lookup", bad_lookup, required=["x"])
        with pytest.raises(ToolExecutionError):
            tool_dispatch.execute("bad_lookup", {"x": "1"}, "")

    def test_connection_error_from_mcp_style_tool_is_wrapped(self, clean_registry):
        def network_call(x):
            raise ConnectionError("upstream unreachable")
        _register("network_call", network_call, required=["x"])
        with pytest.raises(ToolExecutionError):
            tool_dispatch.execute("network_call", {"x": "1"}, "")

    def test_non_string_return_is_wrapped(self, clean_registry):
        def returns_dict(x):
            return {"not": "a string"}
        _register("returns_dict", returns_dict, required=["x"])
        with pytest.raises(ToolExecutionError):
            tool_dispatch.execute("returns_dict", {"x": "1"}, "")

    def test_extra_hallucinated_arguments_are_filtered_not_fatal(self, clean_registry):
        def only_x(x):
            return f"x={x}"
        _register("only_x", only_x, required=["x"])
        events, output = tool_dispatch.execute(
            "only_x", {"x": "real", "y": "hallucinated"}, ""
        )
        assert output == "x=real"

    def test_kwargs_tool_receives_everything(self, clean_registry):
        def flexible(**kwargs):
            return str(sorted(kwargs.items()))
        _register("flexible", flexible, required=[])
        events, output = tool_dispatch.execute("flexible", {"a": "1", "b": "2"}, "")
        assert "'a', '1'" in output.replace('"', "'")


class TestCallSignature:
    def test_same_name_and_args_produce_same_signature(self):
        sig1 = tool_dispatch.call_signature("foo", {"a": 1, "b": 2})
        sig2 = tool_dispatch.call_signature("foo", {"b": 2, "a": 1})
        assert sig1 == sig2

    def test_different_args_produce_different_signature(self):
        sig1 = tool_dispatch.call_signature("foo", {"a": 1})
        sig2 = tool_dispatch.call_signature("foo", {"a": 2})
        assert sig1 != sig2
