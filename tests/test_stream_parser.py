"""StreamParser: the incremental <think>/<tool_call> tag scanner and the
degenerate-reply / foreign-script guards. Fed one delta at a time to mimic
real token-by-token streaming, including deltas that split a tag across a
chunk boundary."""

from src.engine.events import Thought, Token
from src.engine.stream_parser import StreamParser


def run(deltas: list[str], sourced_text: str = ""):
    parser = StreamParser(sourced_text=sourced_text)
    events = []
    for d in deltas:
        events.extend(parser.feed(d))
    trailing, result = parser.finish()
    events.extend(trailing)
    return events, result


def text_of(events):
    return "".join(e.text for e in events if isinstance(e, Token))


class TestPlainText:
    def test_simple_answer(self):
        events, result = run(["Hello", " world"])
        assert text_of(events) == "Hello world"
        assert not result.is_tool_call
        assert not result.degenerate_start
        assert result.full_text == "Hello world"

    def test_tag_split_across_chunks(self):
        # '<think>' arrives split as '<th' + 'ink>' - must not leak '<th'
        # as literal text before the tag is recognized.
        events, result = run(["Hi\n<th", "ink>reasoning</think>", "answer"])
        assert text_of(events) == "Hi\nanswer"
        thoughts = [e for e in events if isinstance(e, Thought)]
        assert thoughts and thoughts[0].text == "reasoning"


class TestThinkBlock:
    def test_think_block_hidden_from_answer(self):
        events, result = run(["<think>secret reasoning</think>The answer"])
        assert "secret reasoning" not in text_of(events)
        assert text_of(events) == "The answer"
        thoughts = [e for e in events if isinstance(e, Thought)]
        assert thoughts[0].text == "secret reasoning"

    def test_empty_think_block_emits_no_thought_event(self):
        # <think></think> (or whitespace-only) carries no information worth
        # showing - the renderer would otherwise print a bare "[thought] "
        # line with nothing after it.
        events, result = run(["<think></think>The answer"])
        assert not [e for e in events if isinstance(e, Thought)]
        assert text_of(events) == "The answer"

    def test_whitespace_only_think_block_emits_no_thought_event(self):
        events, result = run(["<think>   \n  </think>Answer here"])
        assert not [e for e in events if isinstance(e, Thought)]

    def test_unclosed_think_at_eos_is_dropped_not_leaked(self):
        # Regression test for the bug found in the old implementation:
        # generation cut off mid-<think> (hit max_tokens) used to leak the
        # raw, unclosed chain-of-thought straight to the user as if it were
        # the final answer. It must never appear as a Token.
        events, result = run(["<think>half-formed reasoning that never closes"])
        assert text_of(events) == ""
        assert result.full_text == ""
        thoughts = [e for e in events if isinstance(e, Thought)]
        assert thoughts and "half-formed reasoning" in thoughts[0].text


class TestToolCall:
    def test_tool_call_stops_live_text_and_is_classified(self):
        events, result = run([
            "<think>plan</think>",
            '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>',
        ])
        assert text_of(events) == ""
        assert result.is_tool_call
        assert result.tool_call_json == '{"name": "get_weather", "arguments": {}}'

    def test_unclosed_tool_call_still_yields_payload(self):
        """Regression test: the model opened <tool_call> but generation
        stopped before </tool_call>. The JSON payload is usually complete —
        it must reach the caller instead of tool_call_json staying None
        (which fed an empty string to json.loads and gave the model a
        meaningless 'char 0' error to retry from)."""
        events, result = run([
            '<tool_call>\n{"name": "web_search", "arguments": {"query": "mumbai"}}',
        ])
        assert result.is_tool_call
        assert result.tool_call_json is not None
        assert '"web_search"' in result.tool_call_json

    def test_text_before_tool_call_still_flushes(self):
        # Old behavior: any plain text preceding <tool_call> is real content
        # and should still reach the user (rare in practice given the
        # protocol, but the scanner must not special-case it away).
        events, result = run([
            'ok, one sec\n<tool_call>{"name": "x", "arguments": {}}</tool_call>',
        ])
        assert text_of(events) == "ok, one sec\n"
        assert result.is_tool_call


class TestDegenerateReply:
    def test_bare_role_word_is_suppressed(self):
        events, result = run(["user"])
        assert text_of(events) == ""
        assert result.degenerate_start
        assert not result.is_tool_call

    def test_role_word_with_real_content_after_newline_is_suppressed(self):
        events, result = run(["user\nI'm Tuffy, nice to meet you"])
        assert text_of(events) == ""
        assert result.degenerate_start

    def test_real_reply_containing_word_user_is_not_suppressed(self):
        events, result = run(["As a user, you can configure this."])
        assert text_of(events) != ""
        assert not result.degenerate_start

    def test_short_real_reply_not_flagged(self):
        events, result = run(["Yes."])
        assert text_of(events) == "Yes."
        assert not result.degenerate_start

    def test_empty_stream_yields_nothing(self):
        # A genuinely empty stream never reaches the holdback buffer at all
        # (nothing was ever fed), so it isn't classified as degenerate_start
        # here - the caller (turn_engine._handle_final_text) treats a blank
        # full_text as the same "needs a retry/forced answer" case via a
        # separate check, so the net behavior is identical either way.
        events, result = run([])
        assert text_of(events) == ""
        assert result.full_text == ""
        assert not result.is_tool_call


class TestFillerReplies:
    """Regression tests: a small model sometimes emits a literal '...' —
    alone, or as a filler line before its real answer. If that ever gets
    saved as an assistant turn, episodic retrieval re-injects
    'Assistant: ...' into every later prompt and the model learns to answer
    '...' (self-reinforcing loop observed in practice)."""

    def test_bare_dots_reply_is_degenerate(self):
        events, result = run(["..."])
        assert text_of(events) == ""
        assert result.degenerate_start
        assert result.full_text == ""

    def test_dots_with_trailing_whitespace_is_degenerate(self):
        events, result = run(["...  \n"])
        assert text_of(events) == ""
        assert result.degenerate_start

    def test_leading_filler_line_is_stripped_from_real_answer(self):
        events, result = run(["...  \nI'm sorry, but I can't help with that."])
        assert text_of(events) == "I'm sorry, but I can't help with that."
        assert result.full_text == "I'm sorry, but I can't help with that."
        assert not result.degenerate_start

    def test_filler_line_after_think_block_is_stripped(self):
        events, result = run(["<think>the search failed</think>...  \nSorry, the search failed."])
        assert text_of(events) == "Sorry, the search failed."
        assert result.full_text == "Sorry, the search failed."

    def test_inline_ellipsis_in_real_sentence_is_untouched(self):
        events, result = run(["Well... that's a good question, let me think."])
        assert text_of(events) == "Well... that's a good question, let me think."
        assert not result.degenerate_start

    def test_ellipsis_char_reply_is_degenerate(self):
        events, result = run(["…"])
        assert text_of(events) == ""
        assert result.degenerate_start


class TestForeignScript:
    def test_unsourced_foreign_script_is_suppressed(self):
        events, result = run(["Hello\nこんにちは, how are you"])
        assert result.suppressed_foreign

    def test_foreign_script_sourced_from_tool_output_is_allowed(self):
        events, result = run(
            ["Hello\nこんにちは, that's what it means"],
            sourced_text="こんにちは",
        )
        assert not result.suppressed_foreign
        assert "こんにちは" in text_of(events)

    def test_pure_latin_text_never_flagged(self):
        events, result = run(["Bonjour, ça va? Émigré résumé naïve."])
        assert not result.suppressed_foreign
