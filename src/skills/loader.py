"""Skills: droppable capability packs under ./.tuffy/skills/<name>/.

A skill is a directory containing:
  SKILL.md    required. YAML frontmatter (name, description) + a markdown
              body of guidance (when/how to use this skill). Only the
              one-line description is injected into the system prompt by
              default — the full body is fetched on demand via the
              'read_skill' tool, so installing many skills doesn't bloat
              every turn's prompt the way inlining all of them would.
  tools.py    optional. Plain @registry.register(...) functions, exactly
              like src/tools/*.py — auto-imported at startup, so a skill can
              ship its own tools with zero new mechanism to learn.
  mcp.json    optional. One MCP server config this skill wants connected
              (see src/tools/mcp_client.py) — merged into the client's server
              list at startup.

Nothing here executes untrusted code beyond what the user already put in
their own ./.tuffy/skills/ directory — same trust boundary as src/tools/*.py.
"""

import importlib.util
import json
import os

import yaml

# Resolved against this package's own location, not the caller's cwd — the
# terminal always runs with cwd == this repo's root so a relative path was
# invisible historically, but any other consumer importing tuffy as a
# package (e.g. tuffy-ui/backend, started from a different cwd) needs this
# to still find the same .tuffy/skills/. Same fix as src/memory.py's
# DB_DIR and src/tools/mcp_client.py's MCP_CONFIG_PATH.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SKILLS_DIR = os.path.join(_REPO_ROOT, ".tuffy", "skills")

_loaded_skills = {}  # name -> {"description": str, "body": str, "path": str}


def _parse_skill_md(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if text.startswith("---"):
        _, fm_text, body = text.split("---", 2)
        frontmatter = yaml.safe_load(fm_text) or {}
    else:
        frontmatter = {}
        body = text

    return {
        "name": frontmatter.get("name", ""),
        "description": frontmatter.get("description", "").strip(),
        "body": body.strip(),
    }


def discover_skills() -> dict:
    """Scans SKILLS_DIR for skill folders, auto-imports each one's tools.py
    (registering any tools as a side effect), and returns {name: {description,
    body, path}} for every skill with a valid SKILL.md. Safe to call more than
    once (re-scans fresh each time); does not re-import tools.py on repeat
    calls within the same process since Python caches modules by path."""
    global _loaded_skills
    _loaded_skills = {}

    if not os.path.isdir(SKILLS_DIR):
        return _loaded_skills

    for entry in sorted(os.listdir(SKILLS_DIR)):
        skill_dir = os.path.join(SKILLS_DIR, entry)
        skill_md = os.path.join(skill_dir, "SKILL.md")
        if not os.path.isdir(skill_dir) or not os.path.isfile(skill_md):
            continue

        try:
            parsed = _parse_skill_md(skill_md)
        except Exception as e:
            print(f"[skills] Failed to parse {skill_md}: {e}")
            continue

        name = parsed["name"] or entry
        if not parsed["description"]:
            print(f"[skills] Skipping '{name}': SKILL.md has no 'description' in its frontmatter.")
            continue

        tools_path = os.path.join(skill_dir, "tools.py")
        if os.path.isfile(tools_path):
            try:
                spec = importlib.util.spec_from_file_location(f"tuffy_skill_{entry}_tools", tools_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except Exception as e:
                print(f"[skills] Failed to load tools.py for skill '{name}': {e}")

        _loaded_skills[name] = {
            "description": parsed["description"],
            "body": parsed["body"],
            "path": skill_dir,
        }

    return _loaded_skills


def list_skills() -> dict:
    """Currently loaded skills (call discover_skills() first at startup)."""
    return _loaded_skills


def skill_prompt_lines() -> list[str]:
    """One line per skill for the system prompt — name + description only.
    The full guidance body is fetched on demand via the 'read_skill' tool."""
    return [f"- {name}: {info['description']}" for name, info in _loaded_skills.items()]


def mcp_configs_from_skills() -> list[dict]:
    """Collects each skill's optional mcp.json (a single server config dict)
    for merging into the MCP client's server list."""
    configs = []
    for name, info in _loaded_skills.items():
        mcp_json_path = os.path.join(info["path"], "mcp.json")
        if not os.path.isfile(mcp_json_path):
            continue
        try:
            with open(mcp_json_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            config.setdefault("name", f"{name}-skill")
            configs.append(config)
        except Exception as e:
            print(f"[skills] Failed to load mcp.json for skill '{name}': {e}")
    return configs
