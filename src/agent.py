"""The agent orchestration layer: drives the ReAct tool-calling loop against
whichever LLMProvider the active model card resolves to (see src/llm/) —
local llama.cpp or an OpenAI-compatible API. All prompt text (persona,
tool-output framing, error messages) lives in src/prompts/ - this module only
orchestrates, and never talks to llama_cpp or an HTTP client directly."""

import json
import re

from src.memory import add_lesson
from src.llm import build_provider
from src.tools.registry import registry
from src.prompts import templates
from src.vision import IMAGE_SENTINEL

_MAX_TOOL_HOPS = 4

_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

# Scripts the model cannot write reliably itself (Greek/Cyrillic through
# Indic through CJK). Symbols, punctuation and emoji are deliberately outside
# these ranges. Text in these scripts is only trusted when it came out of a
# tool (i.e. the translate tool) this turn.
_FOREIGN_SCRIPT_PATTERN = re.compile(r"[Ͱ-῿⺀-퟿]")


class ToolCallError(Exception):
    """A tool call that couldn't be parsed or executed; carries the tool name
    (when known) so the loop can record a lesson if a retry later succeeds."""

    def __init__(self, message: str, tool_name: str = None):
        super().__init__(message)
        self.tool_name = tool_name


class LocalAgent:
    """Despite the name (kept for compatibility with callers/main.py), this
    now drives ANY provider — local gguf via llama.cpp or a remote API model
    — chosen by the model card's 'provider' field. The ReAct loop, tool
    dispatch, and foreign-script guard below are provider-agnostic; only
    src/llm/*_provider.py contains backend-specific code."""

    def __init__(self, model_card: dict):
        self.model_id = model_card["id"]
        self.sampling_params = model_card["sampling_params"]
        # Optional callable(str) the UI can set to show live status (the
        # model's current thought / which tool is running).
        self.status_cb = None
        # Optional callable(str) for the full ReAct trace (tool calls,
        # arguments, raw results) — set by the CLI only in Ray mode. When
        # None, the agent stays silent about its internal steps; the caller
        # decides what a user is shown, not this module.
        self.trace_cb = None

        self.provider = build_provider(model_card)
        self.provider.load()

    @property
    def supports_vision(self) -> bool:
        return self.provider.supports_vision

    @property
    def vision_disabled_reason(self):
        return self.provider.vision_disabled_reason

    def unload(self):
        """Releases whatever resources the active provider holds (local
        model memory, or just its API credentials) before another model is
        loaded in its place."""
        self.provider.unload()

    @staticmethod
    def attach_image(user_message: dict, image_data_uri: str) -> dict:
        """Rewrites a plain-text user message into the OpenAI-style multimodal
        content list form, appending an image alongside the existing text.
        image_data_uri is a data: URI (base64) or a plain http(s) URL - both
        llama.cpp's MTMDChatHandler and OpenAI-compatible vision APIs accept
        this same content-block shape."""
        return {
            "role": user_message["role"],
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_uri}},
                {"type": "text", "text": user_message["content"]},
            ],
        }

    def _status(self, text: str):
        if self.status_cb is not None and text:
            self.status_cb(text)

    def _trace(self, event: str, **data):
        if self.trace_cb is not None:
            self.trace_cb(event, data)

    def complete(self, **kwargs):
        """Non-streaming completion for internal side-tasks (memory
        reflection, session summaries) that shouldn't touch chat history."""
        return self.provider.complete(**kwargs)

    def run_stream(self, messages: list):
        """The ReAct loop: Thought -> Action (<tool_call>) -> Observation ->
        ... -> final Answer, streamed token by token.

        Loops on tool calls until the model gives a plain-text answer or the
        hop budget runs out. Failed calls come back as error observations so
        the model can self-correct; a correction that then succeeds is saved
        as a lesson for future sessions. A final answer containing non-Latin
        script the model wrote itself (rather than relayed from a tool) is
        intercepted and redirected through the translate tool.
        """
        turn_tool_outputs = []  # everything tools returned this turn, for the script guard
        failed_tools = {}       # tool name -> first error message, for lesson capture
        seen_calls = set()      # (name, sorted-args-json) already executed this turn

        for hop in range(_MAX_TOOL_HOPS):
            is_last_hop = hop == _MAX_TOOL_HOPS - 1
            self._status("thinking")

            response_text, is_tool_call, unsourced_foreign = yield from self._stream_completion(
                messages, sourced_text="".join(turn_tool_outputs)
            )

            if not is_tool_call:
                if unsourced_foreign and not is_last_hop:
                    # The model hand-wrote foreign script — unreliable. Ask it
                    # to route through translate and keep looping.
                    self._status("rewriting via translate")
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({"role": "user", "content": templates.foreign_script_correction()})
                    continue
                return

            tool_call_match = _TOOL_CALL_PATTERN.search(response_text)
            messages.append({"role": "assistant", "content": response_text})

            call_signature = self._call_signature(tool_call_match)
            if call_signature is not None and call_signature in seen_calls:
                # Same tool + same arguments already ran this turn: repeating
                # it can't produce new information (a small model will
                # otherwise spend its whole hop budget re-running e.g.
                # get_system_stats). Tell it plainly instead of executing.
                messages.append({
                    "role": "user",
                    "content": templates.repeated_call_blocked(is_last_hop)
                })
                if is_last_hop:
                    yield from self._final_answer_guaranteed(messages)
                    return
                continue

            try:
                tool_output, function_name = self._execute_tool_call(tool_call_match)
            except ToolCallError as e:
                failed_tools.setdefault(e.tool_name or "?", str(e))
                messages.append({
                    "role": "user",
                    "content": templates.tool_call_failed(str(e), is_last_hop)
                })
                if is_last_hop:
                    yield from self._final_answer_guaranteed(messages)
                    return
                continue

            if call_signature is not None:
                seen_calls.add(call_signature)
            turn_tool_outputs.append(tool_output)
            if function_name in failed_tools:
                # Self-correction succeeded: keep the lesson for next time.
                add_lesson(f"{function_name}: earlier call failed ({failed_tools.pop(function_name)[:120]}); corrected call worked")

            if tool_output.startswith(IMAGE_SENTINEL):
                image_path, _, image_data_uri = tool_output[len(IMAGE_SENTINEL):].partition("\n")
                self._status("analysing image")
                next_step = templates.tool_output_prompt(
                    function_name,
                    f"Image ready and attached below. Saved at: {image_path}. It is already in front of "
                    "you — look at it directly, no further tool call needed to see it.",
                    is_last_hop,
                )
                messages.append(self.attach_image({"role": "user", "content": next_step}, image_data_uri))
            else:
                self._status(f"reading {function_name} result")
                messages.append({
                    "role": "user",
                    "content": templates.tool_output_prompt(function_name, tool_output, is_last_hop)
                })

            if is_last_hop:
                # The hop budget is spent and the last action just succeeded;
                # nothing above forces a reply, so force one now rather than
                # silently ending the turn with no visible answer.
                yield from self._final_answer_guaranteed(messages)
                return

    @staticmethod
    def _call_signature(tool_call_match: re.Match):
        """(name, canonical-args-json) for exact-repeat detection, or None if
        the call doesn't even parse (that's ToolCallError's job to report)."""
        try:
            info = json.loads(tool_call_match.group(1).strip())
            return info.get("name"), json.dumps(info.get("arguments", {}), sort_keys=True)
        except (json.JSONDecodeError, AttributeError, TypeError):
            return None

    def _final_answer_guaranteed(self, messages: list):
        """Forces one last completion with an instruction that makes an
        empty reply impossible, for use when the hop budget runs out. Falls
        back to a fixed sentence if the model still produces nothing."""
        messages.append({"role": "user", "content": templates.force_final_answer()})
        text, _, _ = yield from self._stream_completion(messages)
        if not text.strip():
            fallback = "I wasn't able to finish that with the tools available — could you rephrase or narrow the request?"
            yield fallback

    def _unsourced_foreign(self, text: str, sourced_text: str) -> bool:
        """True when text contains foreign-script characters that did NOT come
        out of a tool this turn (i.e. the model hand-wrote them)."""
        chars = set(_FOREIGN_SCRIPT_PATTERN.findall(text))
        return any(c not in sourced_text for c in chars)

    def _stream_completion(self, messages: list, sourced_text: str = ""):
        """Streams one completion token-by-token as it's generated, using the
        model card's sampling_params (temperature etc.) as-is.

        The system prompt requires tool calls to start with '<tool_call>' and
        contain nothing else, so we only need to buffer long enough to tell
        whether the response is going to be a tool call: if the accumulated
        text so far can still be a prefix of '<tool_call>', hold it back;
        otherwise it's plain text, so flush everything buffered and yield
        the rest live as it streams in.

        Additionally guards against the model hand-writing foreign script it
        can't write reliably: the moment an unsourced foreign character shows
        up, yielding stops (the rest is consumed silently) and the caller is
        told via the third return value so it can redirect through translate.

        Returns (full_text, is_tool_call, unsourced_foreign) via
        StopIteration.value, for callers driving this with `yield from`.
        """
        marker = "<tool_call>"
        buffer = ""
        revealed = False
        suppressed = False
        full_text = ""

        stream = self.provider.stream_completion(messages, **self.sampling_params)
        for chunk in stream:
            delta = chunk["choices"][0]["delta"].get("content")
            if not delta:
                continue
            full_text += delta

            if suppressed:
                continue

            if revealed:
                if self._unsourced_foreign(delta, sourced_text):
                    suppressed = True
                    continue
                yield delta
                continue

            buffer += delta
            stripped = buffer.lstrip()
            if not stripped or marker.startswith(stripped[:len(marker)]):
                # All whitespace so far, or still an unresolved prefix of '<tool_call>' - keep buffering.
                continue

            if self._unsourced_foreign(buffer, sourced_text):
                suppressed = True
                continue

            # Confirmed this isn't a tool call: release everything held back.
            revealed = True
            yield buffer

        if not revealed and not suppressed:
            # Whole response fit in the buffer without resolving - decide now.
            if buffer.strip().startswith(marker):
                return full_text.strip(), True, False
            yield buffer
        return full_text.strip(), False, suppressed

    def _execute_tool_call(self, tool_call_match: re.Match) -> tuple[str, str]:
        """Parses and runs one ReAct action, returning (tool_output, function_name).
        Raises ToolCallError on any failure so the loop can feed it back as an
        error observation."""
        try:
            tool_info = json.loads(tool_call_match.group(1).strip())
        except (json.JSONDecodeError, AttributeError) as e:
            raise ToolCallError(f"tool call is not valid JSON ({e})")

        function_name = tool_info.get("name")
        function_args = tool_info.get("arguments", {}) or {}
        thought = str(tool_info.get("thought", "")).strip()

        if not function_name or function_name in ("tool_name", "exact_tool_name"):
            raise ToolCallError("no real tool name given — use an exact name from the TOOLS list")

        if function_name not in registry.functions:
            raise ToolCallError(
                f"tool '{function_name}' does not exist. Available: {list(registry.functions.keys())}",
                tool_name=function_name,
            )

        missing = [arg for arg in registry.required_args(function_name) if arg not in function_args]
        if missing:
            raise ToolCallError(
                f"missing required argument(s) {missing} for tool '{function_name}'",
                tool_name=function_name,
            )

        self._status(thought or f"using {function_name}")
        self._trace("tool_call", name=function_name, arguments=function_args, thought=thought)

        import inspect
        func = registry.functions[function_name]
        sig = inspect.signature(func)
        has_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())

        if not has_kwargs:
            sanitized_args = {
                k: v for k, v in function_args.items()
                if k in sig.parameters
            }
        else:
            sanitized_args = function_args

        try:
            tool_output = func(**sanitized_args)
        except TypeError as e:
            raise ToolCallError(f"bad arguments for '{function_name}': {e}", tool_name=function_name)

        if tool_output.startswith(IMAGE_SENTINEL):
            image_path = tool_output[len(IMAGE_SENTINEL):].partition("\n")[0]
            shown = f"(image attached, saved at {image_path})"
        else:
            shown = tool_output
        self._trace("tool_result", name=function_name, result=shown[:600])

        return tool_output, function_name
