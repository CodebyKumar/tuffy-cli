# src/

Tuffy's agent core. Nothing in here prints to a terminal or reads stdin — that's `src/cli/`'s
job and `main.py`'s alone. Every other package is usable headless.

| Path | What it owns |
|---|---|
| [agent.py](agent.py) | `LocalAgent` — the ReAct tool-calling loop, provider-agnostic (drives whatever `src/llm/` adapter the active model card resolves to). |
| [identity.py](identity.py) | The agent's fixed self-model (name, capabilities) — never LLM-written, never stored as "memory". |
| [memory.py](memory.py) | Long-term memory (profile/notes/sessions/lessons): the `remember` tool, fact extraction, session summaries, and the quarantine boundary that rejects identity-shaped or junk extractions. |
| [vision.py](vision.py) | Image encoding (file → data URI) and the `IMAGE_SENTINEL` protocol tools use to hand an image back to the agent loop. |
| [cli/](cli/) | The interactive terminal chat: banner, spinner, slash commands, turn loop. Only `main.py` imports this. |
| [llm/](llm/) | `LLMProvider` interface + adapters (`llama_cpp`, `openai_compatible`) — how `agent.py` talks to a model without hardcoding a backend. |
| [models/](models/) | `ModelRegistry` — model cards (local GGUF or API), load params, sampling params; `weights/` holds gitignored GGUF files. |
| [prompts/](prompts/) | Every system-prompt string, centralized: `personas.yaml` (static tone/rules) + `templates.py` (Python-built fragments). |
| [tools/](tools/) | Native tools the agent can call, grouped by domain, plus the MCP client that registers external servers' tools the same way. |
| [skills/](skills/) | Discovery/loading mechanism for `./skills/*/` capability packs (content lives at the repo root, not here). |

## Request flow

1. `main.py` builds the system prompt (`src/prompts.build_system_prompt`) from: identity, tool
   signatures, long-term memory, session summaries, lessons, and installed skills.
2. The user's message goes into `LocalAgent.run_stream()` — a ReAct loop:
   `Thought → <tool_call> JSON → Observation → ... → final text answer`, capped at a fixed hop budget.
3. Tool calls are parsed, dispatched through `src.tools.registry`, and the result fed back as a
   synthetic turn.
4. After the turn, `src/memory.py` runs a small reflection completion to extract durable facts
   about the *user* (never about the agent itself).

See [ARCHITECTURE.md](../ARCHITECTURE.md) at the repo root for the full HLD/LLD design
rationale, including diagrams of the request flow and how each component's contract fits
together.
