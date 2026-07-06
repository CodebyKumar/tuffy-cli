"""Tool package: importing this package registers every native tool as a
side effect. Each domain lives in its own module — see registry.py's GROUP_*
constants for the canonical group names:

  editing.py   - workspace file read/write/list, targeted edit_file
  coding.py    - run_python, run_shell, git status/diff/commit
  research.py  - web_search, translate, get_datetime
  system.py    - get_system_stats, top_processes
  registry.py  - ToolRegistry/decorator machinery only, no tools of its own

(memory.py registers the 'remember' tool itself, group="memory" — it isn't
re-exported here since main.py already imports src.memory directly.)

To add a new tool: pick the module matching its domain (or add a new one),
write a plain function, decorate it with @registry.register(..., group=...),
done — no other file needs to change.
"""

import src.tools.editing  # noqa: F401
import src.tools.coding  # noqa: F401
import src.tools.research  # noqa: F401
import src.tools.system  # noqa: F401
