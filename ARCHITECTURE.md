# Tuffy — Architecture

Tuffy is a personal AI agent built around a provider-agnostic ReAct loop: one conversation loop,
one tool registry, one prompt-building pipeline — all driven through interfaces so the model
backend, the tool set, and the capability surface (skills, MCP servers) can each grow or swap
independently without touching the other two.

This document covers:
1. **High-level design (HLD)** — the system as a whole: major components and how they relate.
2. **Low-level design (LLD)** — each component's internal responsibilities and contracts.
3. **Code structure** — how the design above maps onto `src/`.

For step-by-step configuration (adding a model, connecting an MCP server, writing a skill or a
tool), see [docs/](docs/). For a plain per-folder file listing, see [src/README.md](src/README.md).

---

## 1. High-level design

### 1.1 System overview

```
                              ┌─────────────────────────┐
                              │        main.py           │
                              │  startup wiring + input   │
                              │  loop (src/cli/)          │
                              └────────────┬──────────────┘
                                           │
                     ┌─────────────────────┼─────────────────────┐
                     │                     │                     │
                     ▼                     ▼                     ▼
           ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
           │   Prompt build     │  │   Turn engine       │  │   Session state    │
           │  src/prompts/      │  │   (ReAct loop)      │  │   src/cli/session   │
           │                    │  │   src/engine/        │  │                    │
           └─────────┬──────────┘  └─────────┬──────────┘  └────────────────────┘
                     │                       │
        ┌────────────┼───────────┐           │
        ▼            ▼           ▼           ▼
 ┌────────────┐ ┌───────────┐ ┌────────┐ ┌────────────────┐
 │  identity   │ │  memory    │ │ skills  │ │  LLM provider    │
 │  (fixed)    │ │  (Elastimem)│ │ loader  │ │  src/llm/         │
 └────────────┘ └───────────┘ └────┬───┘ └────────┬─────────┘
                                    │              │
                                    ▼              ▼
                          ┌──────────────┐  ┌───────────────────────┐
                          │  Tool         │  │  Model registry        │
                          │  registry     │  │  src/models/            │
                          │  src/tools/   │  │  (local weights or API) │
                          └──────┬────────┘  └───────────────────────┘
                                 │
                    ┌────────────┼─────────────┐
                    ▼            ▼             ▼
             ┌───────────┐ ┌───────────┐ ┌──────────────┐
             │  Native     │ │  MCP        │ │  Skill-shipped │
             │  tools      │ │  client     │ │  tools          │
             │  (editing,  │ │  (external   │ │  (from          │
             │  coding,    │ │  servers)    │ │  skills/*/       │
             │  research,  │ │              │ │  tools.py)       │
             │  system)    │ │              │ │                  │
             └───────────┘ └───────────┘ └──────────────┘
```

Every arrow into "Tool registry" ends up in the same flat namespace — from the model's point of
view a native tool, an MCP-server tool, and a skill-shipped tool are indistinguishable: same
schema shape, same dispatch call (`registry.functions[name](**args)`).

### 1.2 Core design principles

1. **One interface per swappable axis.** The LLM backend (`LLMProvider`), the tool surface
   (`ToolRegistry`), and the capability-extension mechanisms (skills, MCP) are each behind a
   single seam. The ReAct loop in `src/engine/turn_engine.py` never branches on "which backend"
   or "which kind of tool" — it only calls the interface.
2. **All tools always visible.** There is no runtime mode-switching or per-task tool filtering.
   Every registered tool (native, MCP, or skill-provided) is offered to the model every turn.
   `group` is display metadata only — used for `/tools` output and system-prompt section
   headers — never for hiding a tool.
3. **Identity is fixed; memory is learned.** `src/identity.py` (what the agent *is*: name,
   active model, capabilities) is code-owned and never written by an LLM reflection pass.
   Long-term memory (what the agent has *learned about the user*) lives in the external
   **Elastimem** library, wired in through `src/memory.py`, and is the only part of the system
   prompt an LLM pass can write to — Elastimem's own validation boundary rejects
   identity-shaped keys (`RESERVED_IDENTITY_KEYS`, passed in at `elastimem.open()`) and
   self-referential values, routing rejects to a quarantine log instead of the prompt.
4. **The sandbox is a real boundary.** Every tool that touches the filesystem or runs code
   resolves through `safe_workspace_path()` and is scoped to `agent_workspace/` — not a
   convention, an enforced check that rejects path traversal.
5. **Terminal I/O stays out of the core.** Nothing under `src/engine/`, `src/llm/`,
   `src/tools/`, `src/prompts/`, `src/skills/`, or `src/models/` calls `print()`/`input()`. Only
   `src/cli/` and `main.py` do. This keeps the agent core usable headless behind a different
   frontend later without a rewrite.

---

## 2. Low-level design

### 2.1 Request flow (one user turn)

```
User types a message
        │
        ▼
main.py's input loop ── if it starts with "/" ──▶ src/cli/commands.handle_command()
        │                                                (prints directly, no LLM call)
        │ (else: a normal chat message)
        ▼
src/cli/turn.run_turn()
        │  1. memory.mem.build_context() builds an Elastimem context plan for this input
        │  2. session.system_message(context_plan=plan) rebuilds the system prompt fresh
        │     (src/prompts.build_system_prompt: identity + tools + skills +
        │      memory/episodic sections + session summaries + lessons)
        │  3. appends the user message (with an image attached, if pending)
        │  4. trims history to fit the context budget (src/cli/session.trim_history)
        ▼
src/engine/turn_engine.run_turn() : the ReAct loop, as one flat generator of TurnEvents
        │
        │   for each hop (capped at _MAX_TOOL_HOPS):
        │     stream a completion through the provider (src/llm/*), parsed live by
        │     src/engine/stream_parser.StreamParser into events
        │        │
        │        ├─ plain text  ──▶ Done(full_text): terminal event, loop ends
        │        │
        │        └─ <tool_call> JSON ──▶ tool_dispatch.parse_tool_call() + .execute()
        │                                       │
        │                                       ▼
        │                              ToolResult event; Observation appended as a
        │                              synthetic "user" turn (templates.tool_output_prompt)
        │                                       │
        │                              loop back to the top of the hop
        ▼
src/cli/turn._TurnRenderer renders each event live (spinner label, [thought]/[execute]/[result]
trace lines, streamed answer tokens) as src/cli/turn.py iterates the generator
        │
        ▼
src/cli/turn.py post-processing
        │  - compact_turn(): drops this turn's tool-call/observation
        │    intermediates from history (the final answer already carries
        │    what they contributed)
        │  - memory.mem.record_turn(): hands the turn to Elastimem, which runs a
        │    validated background reflection pass that may write new facts
        │    (never to src/identity.py)
```

A **foreign-script guard** in `src/engine/turn_engine.py` (via `stream_parser`) intercepts
non-Latin text the model hand-writes (unreliable output at small model sizes) mid-stream and
re-routes the turn through the `translate` tool instead of showing unreliable text. A
**degenerate-reply guard** catches a small model's draft answer collapsing into a leaked
chat-role word (e.g. a bare `"user"`) or going empty, and retries instead of showing or saving
garbage; the last hop instead forces one guaranteed final completion
(`_final_answer_guaranteed`) rather than looping forever.

### 2.2 Component contracts

**`LLMProvider`** (`src/llm/base.py`) — the only seam between the ReAct loop and any model
backend:

```
load() / unload()                          lifecycle
complete(**kwargs) -> {...}                 non-streaming, for side-tasks (memory reflection,
                                             session summaries) that shouldn't touch chat history
stream_completion(messages, **params)       yields {"choices":[{"delta":{"content": str}}]}
                                             chunks, shaped like a common chat-completion format
```

Two adapters implement it today: `llama_cpp_provider.py` (local model weights, including a
vision/multimodal path) and `openai_compatible_provider.py` (any HTTP endpoint speaking the
OpenAI chat-completions wire format — this covers most hosted and self-hosted API servers with
no per-provider code). Adding a new backend means writing one more adapter file; the ReAct loop,
tool dispatch, and prompts never change.

**Turn engine** (`src/engine/`) — the ReAct loop itself, expressed as one flat generator of typed
`TurnEvent`s (`events.py`) rather than live-yielded tokens plus a final classification value plus
side-channel callbacks. `turn_engine.run_turn(provider, sampling_params, messages)` is the entry
point; `stream_parser.StreamParser` turns raw text deltas into events (separating `<think>`
blocks, `<tool_call>` JSON, and answer tokens); `tool_dispatch.py` parses and executes tool calls
against `ToolRegistry`; `model_agent.ModelAgent` wraps one `LLMProvider`'s load/unload lifecycle.
Nothing in `src/engine/` imports `src/cli/` — only `src/cli/turn.py` iterates it and renders
events to the terminal.

**`ToolRegistry`** (`src/tools/registry.py`) — the only seam between the ReAct loop and any
callable capability:

```
register(name, description, parameters, required, group) -> decorator
functions: {name: callable}                 dispatch table
schemas: [...]                               OpenAI-function-calling-shaped schemas for the prompt
tool_lines() / tools_by_group()              grouped, human/model-readable views
```

Native tools (`src/tools/*.py`), MCP-server tools (registered dynamically by
`src/tools/mcp_client.py` at startup), and skill-shipped tools (auto-imported by
`src/skills/loader.py`) all funnel into this one registry through the same `register()` call.

**Skills** (`src/skills/loader.py`) — a droppable-folder mechanism, not a code change. A skill
contributes at most three things: a one-line prompt entry (always inlined), a full guidance body
(fetched on demand via the `read_skill` tool, keeping prompt size flat as skills accumulate), and
optionally its own `tools.py`/`mcp.json`.

**MCP client** (`src/tools/mcp_client.py`) — bridges the async-only MCP Python SDK into the
synchronous tool registry via one background thread running its own asyncio event loop. Each
connected server's tools are registered as `<server>_<tool>` under group `mcp:<server>`; a
server that fails to connect is skipped with a warning, never a crash.

**Model registry** (`src/models/registry.py`) — one uniform "model card" shape for both local
and API models (`provider` field selects the adapter; `provider_config` holds API-only fields
like `base_url`/`api_key_env`). `/models` and the switch logic never branch on provider — only
the adapter constructed from the card does.

**Memory** (`src/memory.py`) — a thin adapter wrapping the external **Elastimem** library (a
SQLite-backed facts/episodic/lessons store, `data/memory/tuffy.db`). Two write paths funnel
through it: the `remember`/`recall` tools, and Elastimem's own background worker (fact
extraction, session summarization) fed by `attach_llm()`. `reconfigure_for_model()` must be
called on every model load/switch so Elastimem's context-token budgets track the active model's
real `context_length` instead of staying pinned to the import-time default; `clear_memory()`
(behind `/purge`) archives the DB file and re-attaches the LLM and budgets to a fresh store. See
[data/README.md](data/README.md) for the memory-tier/governor details.

**Identity** (`src/identity.py`) — the opposite of memory: fixed, code-owned, rendered fresh
into every system prompt from the *currently active* model card (so it's always accurate after
a `/models` switch), and structurally incapable of being written by the reflection pass.

### 2.3 Terminal/CLI layer

`src/cli/` is deliberately thin and separated from everything above it:

| Module | Responsibility |
|---|---|
| `display.py` | ANSI colors, the startup logo, the animated status spinner. Pure rendering. |
| `session.py` | `Session` — the active model/agent handle, chat history, history-trimming rules. |
| `commands.py` | One function per slash command + the dispatch table `main.py` calls. |
| `turn.py` | Drives one user turn: builds messages, iterates `src.engine.turn_engine.run_turn()`, renders each `TurnEvent` (spinner/trace/tokens), folds history back together. |

`main.py` itself only does two things: startup wiring (skill discovery, MCP connection) and the
top-level input loop (read a line, dispatch to a command or a chat turn).

---

## 3. Code structure

```
main.py                  Entry point: startup wiring, then the input loop
src/
  cli/                   Terminal chat layer (display, session, commands, turn) — §2.3
  engine/                The ReAct loop itself — §2.1, §2.2
    turn_engine.py          run_turn() — the loop, as a flat generator of TurnEvents
    events.py               Typed events (Status/Thought/ToolCall/ToolResult/Token/Done/Failed)
    stream_parser.py         Raw completion deltas -> events (<think>/<tool_call>/answer split)
    tool_dispatch.py          Parses + executes <tool_call> JSON against ToolRegistry
    model_agent.py             ModelAgent — provider load/unload lifecycle, no ReAct logic
    errors.py                  OutOfMemoryError, ToolExecutionError
  identity.py             Fixed self-model — §2.2
  memory.py              Elastimem adapter: learned long-term memory — §2.2
  settings.py            Persisted user settings (.tuffy/settings.json) — default model id
  vision.py               Image encoding + the IMAGE_SENTINEL hand-off protocol
  llm/
    base.py                LLMProvider interface — §2.2
    llama_cpp_provider.py    Local model-weight backend
    openai_compatible_provider.py  Any OpenAI-wire-format API backend
  models/
    registry.py            ModelRegistry — uniform card shape for local + API models
    __init__.py             DEFAULT_MODEL + imports configs/local.py and configs/api.py
    configs/
      local.py                Local (llama.cpp/gguf) model cards
      api.py                   API-provider (openai_compatible) model cards
    weights/                 Local model weight files (gitignored)
  prompts/
    templates.py            All prompt-string builders
    personas.yaml            Static tone/rules text
    __init__.py              build_system_prompt() — stitches persona + identity + runtime context
  tools/
    registry.py              ToolRegistry — §2.2
    editing.py                File read/write/edit tools, the sandbox boundary
    coding.py                  run_python/run_shell/git tools
    research.py                web_search/get_datetime/translate
    system.py                  get_system_stats/top_processes/capture_image/view_image
    mcp_client.py               MCP client — §2.2
  skills/
    loader.py                Skill discovery/loading — §2.2
    __init__.py                Registers the read_skill tool
.tuffy/
  skills/                 Droppable skill packs (content; the mechanism lives in src/skills/)
  mcp.json                MCP server config (gitignored)
  settings.json            Persisted user settings (gitignored) — e.g. the default model id
docs/                     Configuration how-tos (models, MCP, skills, tools, CLI reference)
data/memory/              Elastimem's SQLite-backed long-term memory store (tuffy.db) + backups/
agent_workspace/          Sandboxed file I/O root every file/code tool is scoped to
```

Every folder above has its own README with implementation-level detail; this document is the
map of *why* it's shaped this way.
