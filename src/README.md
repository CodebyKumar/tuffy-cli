# src/

Tuffy's agent core. Nothing in here prints to a terminal or reads stdin — that's `src/cli/`'s
job and `main.py`'s alone. Every other package is usable headless.

| Path | What it owns |
|---|---|
| [engine/](engine/) | The ReAct loop itself, provider-agnostic: `turn_engine.run_turn()` drives `Thought → <tool_call> → Observation → ...` as one flat generator of typed `TurnEvent`s (see [engine/README.md](engine/README.md) if present, or the module docstrings). Replaces the old `LocalAgent`/`agent.py` design of live-yielded tokens + callbacks. |
| [identity.py](identity.py) | The agent's fixed self-model (name, capabilities) — never LLM-written, never stored as "memory". |
| [memory.py](memory.py) | Thin adapter wrapping the external **Elastimem** library (SQLite-backed facts/episodic/lessons store): the `remember`/`recall` tools, `attach_llm()`/`reconfigure_for_model()` wiring, and `clear_memory()` (backs the `/purge` command). |
| [settings.py](settings.py) | Persisted user settings (`.tuffy/settings.json`, gitignored) — currently just the chosen default model id, set via `/models default <id>`. |
| [vision.py](vision.py) | Image encoding (file → data URI) and the `IMAGE_SENTINEL` protocol tools use to hand an image back to the turn engine. |
| [cli/](cli/) | The interactive terminal chat: banner, spinner, slash commands, turn loop (`turn.py` drives `engine.turn_engine.run_turn` and renders each event). Only `main.py` imports this. |
| [llm/](llm/) | `LLMProvider` interface + adapters (`llama_cpp`, `openai_compatible`) — how the engine talks to a model without hardcoding a backend. API failures (rate limits, bad keys, network errors) surface as `ProviderError`, caught by `src/cli/turn.py` so the session survives. |
| [models/](models/) | `ModelRegistry` — model cards, load params, sampling params, rate-limit metadata. `configs/local.py` and `configs/api.py` hold the actual model card definitions; `weights/` holds gitignored GGUF files. |
| [prompts/](prompts/) | Every system-prompt string, centralized: `personas.yaml` (static tone/rules) + `templates.py` (Python-built fragments). |
| [tools/](tools/) | Native tools the agent can call, grouped by domain, plus the MCP client that registers external servers' tools the same way. |
| [skills/](skills/) | Discovery/loading mechanism for `./.tuffy/skills/*/` capability packs (content lives at the repo root, not here). |

## Environment variables

API-provider models read their key from an environment variable named by the model card's
`api_key_env` (see [llm/README.md](llm/README.md#api-keys)). If it's not exported in your shell,
the provider falls back to reading that one key out of a `.env` file at the repo root — a real
exported env var always wins. `.env` is gitignored, so keys placed there are never committed.

## Request flow

1. `src/cli/turn.py` builds the system prompt fresh (`session.system_message()`, backed by
   `src/prompts.build_system_prompt`) from: identity, tool signatures, long-term memory (an
   Elastimem context plan), session summaries, lessons, and installed skills.
2. The user's message goes into `src.engine.turn_engine.run_turn()` — a ReAct loop:
   `Thought → <tool_call> JSON → Observation → ... → final text answer`, capped at a fixed hop
   budget, expressed as one flat generator of typed events (`Status`, `Thought`, `ToolCall`,
   `ToolResult`, `Token`, `Done`, `Failed`) that `src/cli/turn.py` renders as they arrive.
3. Tool calls are parsed by `src.engine.tool_dispatch`, dispatched through `src.tools.registry`,
   and the result fed back as a synthetic turn.
4. After the turn, Elastimem (via `src/memory.py`'s `attach_llm()`-wired background worker) runs
   fact extraction and session summarization to record durable facts about the *user* (never
   about the agent itself) — see [data/README.md](../data/README.md) for the memory-tier details.

See [ARCHITECTURE.md](../ARCHITECTURE.md) at the repo root for the full HLD/LLD design
rationale, including diagrams of the request flow and how each component's contract fits
together.
