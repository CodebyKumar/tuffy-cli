"""Error taxonomy for the turn engine. Every failure that can end a turn is
one of these, or src.llm.base.ProviderError (the existing contract API
providers already raise for rate limits/network/auth failures) — never a
bare/ambient exception type sniffed by name (the old code guessed
'RuntimeError == llama.cpp OOM', which silently miscategorized any other
RuntimeError raised anywhere in the call tree, including from a tool).
Provider adapters and the tool dispatcher are responsible for translating
whatever they catch into one of these before it leaves them."""


class TurnError(Exception):
    """Base for every turn-ending failure the engine understands."""


class OutOfMemoryError(TurnError):
    """The local backend's decode call failed under memory pressure. Only
    raised by provider adapters that can positively identify this condition
    (see llama_cpp_provider.py) — never inferred from a bare exception type
    at the engine level."""


class ContextOverflowError(TurnError):
    """The assembled prompt (system message + history + this turn's ReAct
    hops) exceeded the model's context window. Distinct from
    OutOfMemoryError: this is a sizing problem the turn can recover from by
    dropping older history, not a resource-pressure problem the machine
    needs to resolve. llama.cpp raises a bare ValueError for this — only the
    provider adapter that can positively identify it is allowed to
    translate it, same rule as OutOfMemoryError above."""


class ToolExecutionError(TurnError):
    """A tool call could not be parsed or its function raised. Carries the
    tool name (when known) so the loop can record a self-correction lesson,
    and is raised for EVERY exception a tool function can produce — not just
    TypeError — so a buggy tool can never take down the session."""

    def __init__(self, message: str, tool_name: str | None = None):
        super().__init__(message)
        self.tool_name = tool_name
