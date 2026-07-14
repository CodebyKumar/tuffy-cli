"""StreamParser: the incremental <think>/<tool_call>/<final_response> tag
scanner and the degenerate-reply / foreign-script guards. Fed one delta at a
time to mimic real token-by-token streaming, including deltas that split a
tag across a chunk boundary."""

from src.engine.events import AnswerStart, Thought, Token
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


class TestFinalResponseTag:
    def test_simple_tagged_answer(self):
        events, result = run(["<think>plan</think><final_response>Hi there</final_response>"])
        assert text_of(events) == "Hi there"
        assert result.full_text == "Hi there"
        assert not result.is_tool_call

    def test_answer_start_fires_before_any_token_of_the_tagged_content(self):
        events, result = run(["<think>plan</think><final_response>Hello world</final_response>"])
        answer_start_idx = next(i for i, e in enumerate(events) if isinstance(e, AnswerStart))
        first_token_idx = next(i for i, e in enumerate(events) if isinstance(e, Token))
        assert answer_start_idx < first_token_idx

    def test_answer_start_fires_exactly_once(self):
        events, result = run(["<final_response>one</final_response>", "trailing stray text"])
        assert sum(1 for e in events if isinstance(e, AnswerStart)) == 1

    def test_tag_split_across_chunks(self):
        # "Hi\n" before the tag still flushes through the old buffered path
        # (see test_content_before_tag_goes_through_old_buffered_path) and
        # is shown live, but full_text (what gets SAVED to history) is
        # authoritatively just the tagged content - this test's focus is
        # that the OPEN tag itself, split across chunk boundaries, is still
        # recognized correctly.
        events, result = run(["Hi\n<final_resp", "onse>answer here</final_resp", "onse>"])
        assert "answer here" in text_of(events)
        assert result.full_text == "answer here"

    def test_multiline_content_with_backslashes_and_pipes_streams_intact(self):
        """Regression test: a live session hit garbled terminal output when
        the model's final answer was a markdown code fence containing
        backslashes and pipe characters (a bird-drawing ASCII-art script) -
        traced to the spinner's own row-clearing logic racing against the
        first Token doubling as the "answer started" signal. The tag +
        AnswerStart fix means this content is never involved in that race:
        confirm it streams through completely unmangled regardless."""
        answer = (
            "```python\n"
            "print('   \\\\|/')\n"
            "print('  /     \\\\\\\\')\n"
            "```\n"
            "This prints a simple bird shape."
        )
        events, result = run([f"<think>writing code</think><final_response>{answer}</final_response>"])
        assert text_of(events) == answer
        assert result.full_text == answer

    def test_closing_tag_split_across_chunks_holds_back_only_the_ambiguous_tail(self):
        # '</final_response>' arriving split must not leak the partial
        # closing-tag characters as literal answer text.
        events, result = run(["<final_response>answer</final_resp", "onse>"])
        assert text_of(events) == "answer"
        assert result.full_text == "answer"

    def test_unclosed_final_response_at_stream_end_still_flushes(self):
        # Generation cut off (max_tokens/EOS) before the closing tag arrived
        # - the tag opened, so this IS the answer; it must not be lost or
        # routed back through degenerate-reply detection.
        events, result = run(["<final_response>partial answer that never closes"])
        assert text_of(events) == "partial answer that never closes"
        assert result.full_text == "partial answer that never closes"
        assert not result.degenerate_start

    def test_untagged_reply_still_works_unchanged(self):
        """Backward compatibility: a model that forgets the tag (small
        models won't comply 100% of the time) must fall back to exactly the
        old untagged behavior, not break or lose the answer."""
        events, result = run(["<think>plan</think>Hi there, no tag used"])
        assert text_of(events) == "Hi there, no tag used"
        assert result.full_text == "Hi there, no tag used"
        assert not any(isinstance(e, AnswerStart) for e in events)

    def test_foreign_script_inside_tag_still_suppressed(self):
        events, result = run(["<final_response>Hello\nこんにちは, how are you</final_response>"])
        assert result.suppressed_foreign

    def test_content_before_tag_goes_through_old_buffered_path(self):
        # Only content INSIDE <final_response> gets the new unbuffered
        # AnswerStart treatment; a stray sentence before the tag (small
        # model preamble) still goes through the existing degenerate-start
        # buffering and is shown live too, but full_text (what gets SAVED)
        # is authoritatively just the tagged content once the tag is used.
        events, result = run(["Sure thing.\n<final_response>Real answer</final_response>"])
        assert "Sure thing." in text_of(events)
        assert "Real answer" in text_of(events)
        assert result.full_text == "Real answer"


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
