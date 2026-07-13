"""The ReAct loop: Thought -> Action (<tool_call>) -> Observation -> ... ->
final Answer, expressed as one flat generator of TurnEvents.

This replaces LocalAgent.run_stream's design of live-yielded tokens plus a
final classification tuple smuggled back through StopIteration.value, driven
by nested `yield from` chains, with side-information (thoughts, tool calls)
pushed out through separate trace_cb/status_cb callbacks. Here everything —
tokens, thoughts, tool calls, tool results, status, and the terminal
outcome — is one ordered sequence of typed events. A caller just does:

    for event in run_turn(provider, sampling_params, messages):
        match event: ...

No generator-internals knowledge required, and no callback wiring needed:
what you get from iterating IS the whole story of the turn.

Error handling: every failure that can end a turn — a local decode OOM, a
provider/network failure, a tool blowing up, the model producing nothing
usable after every guard and retry — becomes a single Failed event with a
`kind` tag. Nothing propagates out of this generator as a raw exception
except a genuine programming bug (which should crash loudly in development,
not be silently swallowed)."""

from src.memory import add_lesson
from src.engine.errors import OutOfMemoryError, ToolExecutionError
from src.engine.events import Done, Failed, Status, ToolResult
from src.engine.stream_parser import StreamParser
from src.engine import tool_dispatch
from src.llm.base import ProviderError
from src.prompts import templates
from src.vision import IMAGE_SENTINEL

_MAX_TOOL_HOPS = 4

_FALLBACK_ANSWER = (
    "I wasn't able to finish that with the tools available — could you "
    "rephrase or narrow the request?"
)


def _stream_one_completion(provider, sampling_params, messages, sourced_text=""):
    """Drives one completion through the StreamParser. Yields events live;
    returns the final ParseResult as the return value (still consumed via
    `yield from` at exactly one call site — run_turn below — kept private to
    this module rather than threaded through multiple frames)."""
    parser = StreamParser(sourced_text=sourced_text)
    stream = provider.stream_completion(messages, **sampling_params)
    for chunk in stream:
        choices = chunk.get("choices")
        if not choices:
            # A trailing usage-only chunk with an empty choices list (some
            # OpenAI-compatible backends emit this) carries no content —
            # skip it instead of indexing into an empty list.
            continue
        delta = choices[0].get("delta", {}).get("content")
        if not delta:
            continue
        for event in parser.feed(delta):
            yield event

    trailing_events, result = parser.finish()
    for event in trailing_events:
        yield event
    return result


def run_turn(provider, sampling_params: dict, messages: list):
    """Runs the full ReAct loop for one user turn against `messages` (mutated
    in place, same contract as the previous implementation — callers pass
    the live session history and get back tool-call scaffolding appended to
    it as hops happen). Always ends in exactly one Done or Failed event."""
    turn_tool_outputs = []
    failed_tools = {}
    seen_calls = set()
    # name -> most recent successful output, for the "you already called
    # this tool once" nudge below. Separate from seen_calls (which is keyed
    # on the FULL signature including arguments) - a model reflexively
    # re-running the same tool with slightly different arguments (a
    # reworded search query, a different-but-equivalent parameter) sails
    # right past the exact-signature guard even though it's just as
    # pointless as a literal repeat, since the first result usually already
    # answers the question.
    tool_last_output: dict[str, str] = {}

    try:
        for hop in range(_MAX_TOOL_HOPS):
            is_last_hop = hop == _MAX_TOOL_HOPS - 1
            yield Status("thinking")

            result = yield from _stream_one_completion(
                provider, sampling_params, messages,
                sourced_text="".join(turn_tool_outputs),
            )

            if not result.is_tool_call:
                outcome = _handle_final_text(result, is_last_hop)
                if outcome == "return_done":
                    yield Done(result.full_text)
                    return
                if outcome == "retry_degenerate":
                    yield Status("retrying")
                    messages.append({"role": "user", "content": templates.degenerate_reply_correction()})
                    continue
                if outcome == "retry_foreign":
                    yield Status("rewriting via translate")
                    messages.append({"role": "assistant", "content": result.full_text})
                    messages.append({"role": "user", "content": templates.foreign_script_correction()})
                    continue
                if outcome == "force_final":
                    text = yield from _final_answer_guaranteed(provider, sampling_params, messages)
                    yield Done(text)
                    return
                # outcome == "return_done_empty": belt-and-suspenders, treat
                # as a real (if blank-ish) answer rather than looping forever
                yield Done(result.full_text)
                return

            messages.append({"role": "assistant", "content": result.full_text})

            try:
                function_name, function_args, thought = tool_dispatch.parse_tool_call(result.tool_call_json or "")
            except ToolExecutionError as e:
                yield from _handle_tool_failure(str(e), None, is_last_hop, messages, failed_tools)
                if is_last_hop:
                    text = yield from _final_answer_guaranteed(provider, sampling_params, messages)
                    yield Done(text)
                    return
                continue

            signature = tool_dispatch.call_signature(function_name, function_args)
            if signature in seen_calls:
                messages.append({
                    "role": "user",
                    "content": templates.repeated_call_blocked(is_last_hop),
                })
                if is_last_hop:
                    text = yield from _final_answer_guaranteed(provider, sampling_params, messages)
                    yield Done(text)
                    return
                continue

            # Not an exact repeat (that's blocked above), but the same tool
            # NAME already ran this turn with different arguments - a
            # genuinely different second use is legitimate (two different
            # searches, a different offset), so the call still runs; it's
            # only reminded, via the Observation appended after execution
            # below, what the first call already returned - so it can't
            # "forget" the first result and treat a near-duplicate call as
            # if nothing had happened.
            already_called_reminder = tool_last_output.get(function_name)

            yield Status(f"using {function_name}")
            try:
                tool_events, tool_output = tool_dispatch.execute(function_name, function_args, thought)
            except ToolExecutionError as e:
                yield from _handle_tool_failure(str(e), e.tool_name, is_last_hop, messages, failed_tools)
                if is_last_hop:
                    text = yield from _final_answer_guaranteed(provider, sampling_params, messages)
                    yield Done(text)
                    return
                continue

            for event in tool_events:
                yield event

            seen_calls.add(signature)
            tool_last_output[function_name] = tool_output
            turn_tool_outputs.append(tool_output)
            if function_name in failed_tools:
                add_lesson(
                    f"{function_name}: earlier call failed "
                    f"({failed_tools.pop(function_name)[:120]}); corrected call worked"
                )

            reminder_prefix = (
                templates.same_tool_called_again_prefix(function_name, already_called_reminder)
                if already_called_reminder is not None else ""
            )

            if tool_output.startswith(IMAGE_SENTINEL):
                image_path, _, image_data_uri = tool_output[len(IMAGE_SENTINEL):].partition("\n")
                yield Status("analysing image")
                next_step = reminder_prefix + templates.tool_output_prompt(
                    function_name,
                    f"Image ready and attached below. Saved at: {image_path}. It is already in front of "
                    "you — look at it directly, no further tool call needed to see it.",
                    is_last_hop,
                )
                messages.append(_attach_image({"role": "user", "content": next_step}, image_data_uri))
            else:
                yield Status(f"reading {function_name} result")
                messages.append({
                    "role": "user",
                    "content": reminder_prefix + templates.tool_output_prompt(function_name, tool_output, is_last_hop),
                })

            if is_last_hop:
                text = yield from _final_answer_guaranteed(provider, sampling_params, messages)
                yield Done(text)
                return

    except OutOfMemoryError as e:
        yield Failed(kind="oom", message=str(e))
    except ProviderError as e:
        yield Failed(kind="provider", message=str(e))
    except GeneratorExit:
        raise
    except KeyboardInterrupt:
        yield Failed(kind="interrupted", message="Interrupted")


def _handle_final_text(result, is_last_hop: bool) -> str:
    """Classifies a non-tool-call completion into what run_turn should do
    next. Pure function, no side effects, so it's trivially unit-testable."""
    if result.degenerate_start:
        return "force_final" if is_last_hop else "retry_degenerate"
    if result.suppressed_foreign:
        return "force_final" if is_last_hop else "retry_foreign"
    if not result.full_text.strip():
        return "force_final" if is_last_hop else "retry_degenerate"
    return "return_done"


def _handle_tool_failure(error_message: str, tool_name, is_last_hop: bool,
                          messages: list, failed_tools: dict):
    if tool_name:
        failed_tools.setdefault(tool_name, error_message)
    yield ToolResult(name=tool_name or "?", result=error_message, ok=False)
    messages.append({
        "role": "user",
        "content": templates.tool_call_failed(error_message, is_last_hop),
    })


def _final_answer_guaranteed(provider, sampling_params, messages: list):
    """Forces one last completion with an instruction that makes an empty
    reply impossible. Falls back to a fixed sentence if the model still
    produces nothing — that fallback is returned as the answer text (the
    caller appends it to history), matching the previous implementation's
    behavior of standing in for the model rather than ending the turn with
    literally nothing shown."""
    messages.append({"role": "user", "content": templates.force_final_answer()})
    result = yield from _stream_one_completion(provider, sampling_params, messages)
    if not result.full_text.strip() or result.degenerate_start:
        return _FALLBACK_ANSWER
    return result.full_text


def _attach_image(user_message: dict, image_data_uri: str) -> dict:
    return {
        "role": user_message["role"],
        "content": [
            {"type": "image_url", "image_url": {"url": image_data_uri}},
            {"type": "text", "text": user_message["content"]},
        ],
    }
