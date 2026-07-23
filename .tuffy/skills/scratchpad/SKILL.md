---
name: scratchpad
description: Jot, list, and clear short-lived timestamped notes for the current project — a scratch task list that survives across turns and sessions without going into long-term memory.
---

# Scratchpad

Use this when the user asks you to jot something down, keep a running list of TODOs, or wants
to check what's been noted so far — anything that's project-scratch, not a durable fact about
the user (that belongs in long-term memory instead, via `remember`).

1. To add a note: call `scratch_note` with the text. Each note is stored with a timestamp.
2. To see what's been noted: call `scratch_list`. Returns all notes, oldest first.
3. To clear everything: call `scratch_clear`. This is destructive — confirm with the user first
   unless they explicitly asked to clear/reset the scratchpad.

Notes are stored per-project (in this skill's own folder), not per-conversation — they persist
across `/new` and `/clear` and are visible next session too.
