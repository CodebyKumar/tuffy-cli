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
           │   Prompt build     │  │   Agent (ReAct)    │  │   Session state    │
           │  src/prompts/      │  │   loop             │  │   src/cli/session   │
           │                    │  │   src/agent.py      │  │                    │
           └─────────┬──────────┘  └─────────┬──────────┘  └────────────────────┘
                     │                       │
        ┌────────────┼───────────┐           │
        ▼            ▼           ▼           ▼
 ┌────────────┐ ┌───────────┐ ┌────────┐ ┌────────────────┐
 │  identity   │ │  memory    │ │ skills  │ │  LLM provider    │
 │  (fixed)    │ │  (learned) │ │ loader  │ │  src/llm/         │
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
   single seam. The ReAct loop in `src/agent.py` never branches on "which backend" or "which
   kind of tool" — it only calls the interface.
2. **All tools always visible.** There is no runtime mode-switching or per-task tool filtering.
   Every registered tool (native, MCP, or skill-provided) is offered to the model every turn.
   `group` is display metadata only — used for `/tools` output and system-prompt section
   headers — never for hiding a tool.
3. **Identity is fixed; memory is learned.** `src/identity.py` (what the agent *is*: name,
   active model, capabilities) is code-owned and never written by an LLM reflection pass.
   `src/memory.py` (what the agent has *learned about the user*) is the only part of the system
   prompt an LLM pass can write to, and it runs through a validation boundary that actively
   rejects identity-shaped keys and values.
4. **The sandbox is a real boundary.** Every tool that touches the filesystem or runs code
   resolves through `safe_workspace_path()` and is scoped to `agent_workspace/` — not a
   convention, an enforced check that rejects path traversal.
5. **Terminal I/O stays out of the core.** Nothing under `src/agent.py`, `src/llm/`,
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
        │  1. session.system_message() rebuilds the system prompt fresh
        │     (src/prompts.build_system_prompt: identity + tools + skills +
        │      memory + session summaries + lessons)
        │  2. appends the user message (with an image attached, if pending)
        │  3. trims history to fit the context budget (src/cli/session.trim_history)
        ▼
src/agent.py — LocalAgent.run_stream() : the ReAct loop
        │
        │   for each hop (capped at _MAX_TOOL_HOPS):
        │     stream a completion through self.provider (src/llm/*)
        │        │
        │        ├─ plain text  ──▶ done: yield tokens to the terminal, return
        │        │
        │        └─ <tool_call> JSON ──▶ parse → registry.functions[name](**args)
        │                                       │
        │                                       ▼
        │                              Observation appended as a synthetic
        │                              "user" turn (templates.tool_output_prompt)
        │                                       │
        │                              loop back to the top of the hop
        ▼
Final answer streamed to the terminal
        │
        ▼
src/cli/turn.py post-processing
        │  - compact_turn(): drops this turn's tool-call/observation
        │    intermediates from history (the final answer already carries
        │    what they contributed)
        │  - extract_facts(): a validated reflection pass that may write
        │    new facts to src/memory.py (never to src/identity.py)
```

A **foreign-script guard** in `src/agent.py` intercepts non-Latin text the model hand-writes
(unreliable output at small model sizes) mid-stream and re-routes the turn through the
`translate` tool instead of showing unreliable text.

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

**Memory** (`src/memory.py`) — three write paths (the `remember` tool, a post-turn reflection
pass, and session/lesson logging on exit), one validation boundary: `store_fact()` rejects
identity-shaped keys (`RESERVED_IDENTITY_KEYS`) and self-referential values
(`is_self_referential_value()`), routing rejects to `quarantine.json` instead of the prompt.

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
| `turn.py` | Drives one user turn: builds messages, streams tokens, folds history back together. |

`main.py` itself only does two things: startup wiring (skill discovery, MCP connection) and the
top-level input loop (read a line, dispatch to a command or a chat turn).

---

## 3. Code structure

```
main.py                  Entry point: startup wiring, then the input loop
src/
  cli/                   Terminal chat layer (display, session, commands, turn) — §2.3
  agent.py               The ReAct loop itself — §2.1
  identity.py             Fixed self-model — §2.2
  memory.py              Learned long-term memory — §2.2
  vision.py               Image encoding + the IMAGE_SENTINEL hand-off protocol
  llm/
    base.py                LLMProvider interface — §2.2
    llama_cpp_provider.py    Local model-weight backend
    openai_compatible_provider.py  Any OpenAI-wire-format API backend
  models/
    registry.py            ModelRegistry — uniform card shape for local + API models
    __init__.py             Where models actually get registered
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
skills/                   Droppable skill packs (content; the mechanism lives in src/skills/)
docs/                     Configuration how-tos (models, MCP, skills, tools, CLI reference)
data/memory/              JSON-backed long-term memory store
agent_workspace/          Sandboxed file I/O root every file/code tool is scoped to
```

Every folder above has its own README with implementation-level detail; this document is the
map of *why* it's shaped this way.
