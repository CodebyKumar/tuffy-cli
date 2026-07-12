# Handoff: Replace Tuffy's `src/memory.py` with Elastimem

Instructions for an AI agent (or human) to swap Tuffy's JSON-file memory for
the Elastimem framework at `~/Projects/elastimem`. Written 2026-07-12 by the
session that built both. Read this whole file before editing anything.

## Background — what exists on each side

**Tuffy** (this repo) is a local agent: `main.py` REPL → `src/cli/turn.py run_turn()` → `src/agent.py LocalAgent.run_stream()` (ReAct loop, llama.cpp,
Qwen3-VL 2B, `n_ctx=4096`, 8 GB Apple Silicon / Jetson Orin).
Current memory = `src/memory.py`: flat JSON under `data/memory/`
(`user/profile.json`, `user/notes.json`, `sessions/log.json`,
`agent/lessons.json`, `quarantine.json`). Public API used elsewhere:
`load_memory, store_fact, load_sessions, add_session_summary, load_lessons, add_lesson, load_quarantine, clear_memory, extract_facts, summarize_session`
plus the registered `remember` tool. Known callers:

- `src/prompts/__init__.py` `build_system_prompt()` → calls `load_memory()`,
  `load_sessions()`, `load_lessons()`, renders via
  `src/prompts/templates.py runtime_context(...)`.
- `src/cli/turn.py` — calls `extract_facts(...)` **synchronously** after each
  turn (~line 80), and uses `trim_history`/`compact_turn` from
  `src/cli/session.py` (`MAX_HISTORY_CHARS = 6000`).
- `src/cli/session.py Session.end()` — `summarize_session` + `add_session_summary`.
- `src/agent.py` — `add_lesson(...)` when a failed tool later succeeds.
- `src/cli/commands.py` — `/memory` (shows facts/sessions/lessons), `/clear`
  (wipes memory + chat), `/new` (chat only).
- `src/identity.py` — `RESERVED_IDENTITY_KEYS` (identity keys memory must
  refuse); keep this module untouched.

**Elastimem** (`~/Projects/elastimem`, importable package `elastimem`, 66
passing tests) is a host-agnostic memory engine: single SQLite file, Memory
Governor (LITE/STANDARD/FULL tiers from RAM, token budgets from n_ctx),
temporally versioned facts, episodic transcripts + FTS5(+vector) recall,
background worker with a foreground-wins LLM gate. Read
`~/Projects/elastimem/docs/integration.md` and `docs/api.md` first.
Key API:

```python
import elastimem
mem = elastimem.open(path, llm=fn, embedder=fn2, context_tokens=4096,
                     static_prompt_tokens=N, reserved_keys={...})
# llm signature: (prompt: str, *, max_tokens: int, temperature: float) -> str
mem.tick(); mem.report_pressure()
plan = mem.build_context(user_input)   # ContextPlan: .sections dict
#   section keys: "user_facts", "relevant_past_moments",
#                 "previous_sessions", "lessons"; also .rolling_summary,
#                 .keep_last_n_turns, .render()
with mem.foreground(): ...             # bracket ALL chat-model generation
mem.record_turn(user, reply)           # persist + rule capture + bg extraction
mem.report_evictions([(u, a), ...])    # pairs trimmed from the live window
mem.remember(key, value)               # -> (changed, reason)
mem.recall(query, k=5)                 # -> [Hit(kind,text,date,score)]
mem.add_lesson(text); mem.lessons()
mem.sessions(); mem.resume_session(); mem.fact_history(key); mem.forget(key)
mem.quarantine_entries(); mem.stats(); mem.end_session(); mem.close()
```

## Step 1 — dependency

```bash
cd ~/Projects/tuffy && uv add --editable ~/Projects/elastimem
```

(Editable path dep is intentional; both repos are local. Verify:
`uv run python -c "import elastimem; print(elastimem.__version__)"`.)

## Step 2 — new `src/memory.py` (thin adapter, keep the module path)

Replace the file's contents entirely. It must:

1. **Create the singleton store** at import time (module-level `mem`),
   DB path `./data/memory/tuffy.db`:
   ```python
   import elastimem
   from src.identity import RESERVED_IDENTITY_KEYS
   mem = elastimem.open(
       "./data/memory/tuffy.db",
       context_tokens=4096,          # read from active model card if easy
       reserved_keys=frozenset(RESERVED_IDENTITY_KEYS),
   )
   ```

   Do **not** pass `llm=` here — the model isn't loaded at import time.
   Instead expose `def attach_llm(complete_fn): mem.complete_fn = _adapt(complete_fn)`
   and call it from `main.py` after the agent loads.
2. **Adapter for the LLM signature.** Tuffy's `LocalAgent.complete(**kw)`
   takes `messages=[...]` and returns a llama.cpp response dict. Elastimem
   wants `(prompt, *, max_tokens, temperature) -> str`. Write:
   ```python
   def _adapt(agent_complete):
       def llm(prompt, *, max_tokens, temperature):
           r = agent_complete(messages=[{"role": "user", "content": prompt}],
                              max_tokens=max_tokens, temperature=temperature)
           return r["choices"][0]["message"]["content"]
       return llm
   ```
3. **Honor `TUFFY_NO_AUTO_MEMORY=1`**: if set, never attach the llm (rule
   capture + explicit remember still work).
4. **One-time JSON migration** (run at import if `tuffy.db` was just created
   and old JSON files exist): read the four legacy JSON files with
   `json.load`, then `mem.remember(k, v, source="import")` for each
   profile/notes entry, `mem.add_lesson(t)` for lessons; move the JSON files
   to `data/memory/legacy-json.bak/`. Session summaries may be skipped (low
   value) or inserted via mem's `sessions` table if you want fidelity.
5. **Re-export the old API names** so callers keep working:
   `load_memory()` → `mem.facts()`; `store_fact(k, v, source="explicit")` →
   `mem.remember(...)`; `load_sessions(n=3)` → `[s["summary"] for s in mem.sessions(n) if s["summary"]][::-1]`; `add_lesson` → `mem.add_lesson`;
   `load_lessons()` → `mem.lessons()`; `load_quarantine(n)` →
   `mem.quarantine_entries(n)`; `clear_memory()` → close db, archive file,
   reopen (or expose and call a wipe); `extract_facts(...)` → **no-op stub**
   returning `{}` (extraction is now background inside `record_turn`);
   `summarize_session(...)` → no-op `""` (handled by `mem.end_session()`).
6. **Keep the `remember` tool registration** (same
   `@registry.register(...)` block as the old file, body =
   `changed, reason = mem.remember(key, value)`), and **add a new
   `memory_search` tool**: body formats `mem.recall(query)` hits as
   `- [date] text` lines (return "nothing found" if empty). Import
   `src.tools.registry` exactly as the old file did.

## Step 3 — wire the turn loop (`src/cli/turn.py`)

- Start of turn: `mem.tick()`.
- Delete the synchronous `extract_facts(...)` call.
- After the final answer is streamed and `compact_turn()` has run:
  `mem.record_turn(user_text, final_answer_text)`.
- Wrap the streaming generation (`session.agent.run_stream(...)` loop) in
  `with mem.foreground():` — this is the single-model-instance safety gate.
- Where llama.cpp decode errors are caught (RuntimeError around the stream),
  add `mem.report_pressure()`.

## Step 4 — prompt building (`src/prompts/__init__.py` + `templates.py`)

`build_system_prompt` currently fetches memory itself. Change it to accept an
optional `context_plan` (an Elastimem `ContextPlan`); `Session.system_message()`
should call `mem.build_context(pending_user_input)` and pass it through.
Map plan sections into `runtime_context(...)`:

- `plan.sections["user_facts"]` → the WHAT YOU KNOW ABOUT THE USER block
  (pass the pre-rendered string; adjust `runtime_context` to accept strings
  instead of dicts/lists where needed).
- `plan.sections["previous_sessions"]` → PREVIOUS SESSIONS.
- `plan.sections["lessons"]` → LESSONS block.
- `plan.sections["relevant_past_moments"]` → NEW section "RELEVANT PAST
  MOMENTS (from your memory of previous conversations)" — add it to
  `templates.runtime_context`.
- If `plan.rolling_summary`, append an "EARLIER IN THIS CONVERSATION
  (condensed): ..." line.
  Threading note: `run_turn` must obtain the plan **before** appending the new
  user message and rebuild `messages[0]` from it (same place it currently calls
  `session.system_message()` — pass the user input through).

## Step 5 — window management (`src/cli/session.py`)

- Keep `compact_turn` and `keep_only_latest_image` unchanged.
- Replace `trim_history` internals: evict oldest user/assistant *pairs*
  beyond `plan.keep_last_n_turns` (never the newest user msg, never
  image-bearing messages), collect evicted `(user, assistant)` text pairs,
  call `mem.report_evictions(pairs)`. Delete `MAX_HISTORY_CHARS`.
- `Session.end()`: replace summarize/add_session_summary with
  `mem.end_session()`; `main.py` shutdown should call `mem.close()` **before**
  `os._exit` (worker drain ≤5 s).
- Model switch (`/models` handling): call `mem.drain()` before
  `agent.unload()`, and re-attach the adapted llm after the new model loads.

## Step 6 — CLI (`src/cli/commands.py`)

- `/memory` → show `mem.stats()` one-liner (tier, counts) + `mem.facts()` +
  recent `mem.sessions()` summaries + `mem.lessons()`.
- Add `/memory search <q>` (recall hits), `/memory facts <key>`
  (`fact_history` version chain), `/memory forget <key>`, `/memory quarantine`.
- `/clear` semantics: keep Tuffy's current behavior (wipe memory + chat) OR
  split into `/clear` = chat only, `/memory clear` = archive DB — prefer the
  split; confirm with the user if unsure.
- Optional: `/sessions` list + `/resume` using `mem.resume_session()`.

## Step 7 — embeddings (OPTIONAL, do last or skip)

FTS5-only recall works fine. If adding vectors: download a MiniLM-class
embedding GGUF, load a **separate** `Llama(model_path=..., embedding=True, n_ctx=512, n_gpu_layers=0)` (CPU — never contend with the chat model; see
memory note: Metal + llama.cpp is fragile on this machine), pass
`embedder=lambda texts: [e.create_embedding(t)["data"][0]["embedding"] for t in texts]`.

## Verification (do all)

1. `uv run pytest` in `~/Projects/elastimem` still green (66 tests).
2. Fresh start: delete `data/memory/`, run tuffy, say "my name is <X></x>, I live
   in <Y></y>" → `/memory` shows both (rule capture, no LLM needed).
3. Migration: restore old JSON files, delete `tuffy.db`, boot → old facts
   appear in `/memory`; JSON moved to `legacy-json.bak/`.
4. Flagship: session 1 "my car needs new brake pads", exit (clean exit writes
   session summary). Session 2: "what did we discuss about my car" → the
   RELEVANT PAST MOMENTS section (or `memory_search` tool) surfaces it.
5. Latency: replies stream immediately; extraction no longer blocks between
   turns (it was synchronous before — this should feel faster).
6. 20+ turn conversation: no context overflow (decode -3 = OOM per memory
   notes), old turns evicted + rolling summary/marker appears.
7. Exit cleanly (`/exit` or Ctrl-D): no hang (worker drains ≤5 s), reboot
   shows the session in `/memory` with a summary.
8. `TUFFY_NO_AUTO_MEMORY=1` still suppresses all background LLM memory calls.

## Constraints & gotchas

- **Never call the chat model from two threads.** The `foreground()` gate in
  step 3 is mandatory; Elastimem's worker respects it.
- `remember` tool responses must stay user-friendly strings (old format:
  `"Stored 'k' = 'v' in long-term memory."` / `"Didn't store it: <reason>."`).
- Commit messages: plain, **no Co-Authored-By trailers** (user preference).
- Do not publish Elastimem to PyPI; do not touch README/docs of tuffy unless
  asked (user preference).
- `main.py` uses `os._exit()` to dodge llama.cpp destructor crashes — keep
  that, just `mem.close()` first.
