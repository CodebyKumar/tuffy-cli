# src/engine/

The ReAct loop, provider-agnostic and terminal-agnostic. Nothing here talks to a Session, a
callback, or a terminal — it only consumes an `LLMProvider` and a message list, and produces a
flat stream of typed events. Replaces the old `src/agent.py` (`LocalAgent.run_stream`), which
mixed live-yielded tokens, a final classification tuple smuggled through
`StopIteration.value`, and side-channel `trace_cb`/`status_cb` callbacks into one hard-to-follow
generator.

- [turn_engine.py](turn_engine.py) - `run_turn(provider, sampling_params, messages)`: the loop
  itself — `Thought → <tool_call> JSON → Observation → ... → final answer`, capped at
  `_MAX_TOOL_HOPS`. Handles degenerate/empty replies, foreign-script leaks, repeated tool calls,
  and tool failures, always ending in exactly one `Done` or `Failed` event.
- [events.py](events.py) - the typed events themselves (`Status`, `Thought`, `ToolCall`,
  `ToolResult`, `Token`, `Done`, `Failed`). A caller just does
  `for event in run_turn(...): match event: ...` — no generator-internals knowledge required.
- [stream_parser.py](stream_parser.py) - `StreamParser`: turns raw completion-stream text deltas
  into events, separating `<think>` blocks, `<tool_call>` JSON, and plain answer tokens; recovers
  truncated `<tool_call>` blocks missing a closing tag and drops empty `<think>` blocks.
- [tool_dispatch.py](tool_dispatch.py) - parses a `<tool_call>` JSON payload (tolerating
  prefix/trailing chatter around the object), computes a call signature for repeat-detection, and
  executes against `src.tools.registry`.
- [model_agent.py](model_agent.py) - `ModelAgent`: thin wrapper owning an `LLMProvider`'s
  load/unload lifecycle, vision capability flags, and the non-streaming `complete()` used by
  memory's background jobs. Holds no ReAct logic itself.
- [errors.py](errors.py) - `OutOfMemoryError`, `ToolExecutionError` — typed exceptions
  `turn_engine` catches and turns into `Failed` events.

## Why this is split out

`src/cli/turn.py` is the only consumer today (it iterates `turn_engine.run_turn()` and renders
each event to the terminal), but nothing in this package imports anything from `src/cli/` —
keeping the loop usable behind a future non-terminal frontend without a rewrite.
