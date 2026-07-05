"""Every prompt string built in Python (as opposed to the static persona text
in personas.yaml) lives here: the self-model and runtime context blocks
appended to the system prompt, the generic ReAct observation message, and the
tool-failure message.

Nothing in src/agent.py should contain a raw prompt string — if the model
needs to be told something, it's a template function in this file.
"""

from datetime import datetime

from src import identity

# Three canonical, tool-agnostic ReAct traces. Deliberately generic — real
# tool-specific values in examples get parroted back by small models as if
# they were real input (an earlier example literally caused the agent to
# view_image('/Users/me/Desktop/screenshot.png')).
_REACT_EXAMPLES = """\
Example — no tool needed:
User: hey, how's it going?
Assistant: All good — what can I do for you?

Example — one tool:
User: <something you need a tool for>
Assistant: <tool_call>
{"thought": "<one short line: what the user needs -> which tool>", "name": "<tool name from the list>", "arguments": {"<arg>": "<value taken from the user's actual message>"}}
</tool_call>
(after the Observation comes back)
Assistant: <final answer grounded in the observation>

Example — chained tools:
User: <something needing two steps, e.g. look something up then translate it>
Assistant: <tool_call>
{"thought": "first I need X", "name": "<tool A>", "arguments": {...}}
</tool_call>
(Observation A comes back)
Assistant: <tool_call>
{"thought": "now convert/refine with tool B", "name": "<tool B>", "arguments": {...}}
</tool_call>
(Observation B comes back)
Assistant: <final answer>"""


def self_model(model_card: dict) -> str:
    """The agent's model of itself: identity (fixed, from src/identity.py),
    plus when 'now' is. Injected every turn so identity questions get real
    answers without the model needing to guess or "remember" who it is."""
    now = datetime.now().strftime("%A %Y-%m-%d, %H:%M")
    return (
        "WHO YOU ARE RIGHT NOW\n"
        f"- Current date/time: {now}.\n"
        f"{identity.describe(model_card)}\n"
        "- Your tools, memory, and past-session summaries are listed below — that IS "
        "what you know and can do; answer questions about yourself from it directly."
    )


def runtime_context(
    tool_lines: list,
    known_facts: dict,
    session_summaries: list,
    lessons: list,
) -> str:
    """The dynamic part of the system prompt: tool signatures, long-term
    memory, episodic session summaries, and self-learned lessons."""
    sections = []

    sections.append("TOOLS YOU CAN CALL\n" + "\n".join(tool_lines))

    if known_facts:
        facts_block = "\n".join(f"- {k}: {v}" for k, v in known_facts.items())
        sections.append(
            "WHAT YOU KNOW ABOUT THE USER (long-term memory — answer from this "
            "directly, no tool call needed):\n" + facts_block
        )
    else:
        sections.append(
            "WHAT YOU KNOW ABOUT THE USER: nothing stored yet. As you learn "
            "facts about them, they will appear here."
        )

    if session_summaries:
        sections.append(
            "PREVIOUS SESSIONS (most recent last):\n"
            + "\n".join(f"- {s}" for s in session_summaries)
        )

    if lessons:
        sections.append(
            "LESSONS FROM YOUR OWN PAST MISTAKES (apply these):\n"
            + "\n".join(f"- {l}" for l in lessons)
        )

    sections.append("PROTOCOL EXAMPLES\n" + _REACT_EXAMPLES)

    return "\n\n".join(sections)


def tool_output_prompt(function_name: str, tool_output: str, is_last_hop: bool) -> str:
    """The ReAct Observation message appended after a tool call succeeds."""
    if is_last_hop:
        next_step = (
            "You have no tool calls left. Give your final answer now, in plain text, "
            "grounded in this observation — respond to what the user actually asked, "
            "don't just repeat the observation back."
        )
    else:
        next_step = (
            "Decide: if you still need another step, respond with ONLY the next "
            "<tool_call> (with a fresh thought). Otherwise give your final answer in "
            "plain text, grounded in this observation — respond to what the user "
            "actually asked, don't just repeat the observation back."
        )
    return f"Observation from {function_name}:\n{tool_output}\n\n{next_step}"


def tool_call_failed(error: str, is_last_hop: bool = True) -> str:
    """Observation message when a <tool_call> fails to parse or execute."""
    if is_last_hop:
        return (
            f"Observation: your tool call failed — {error}. You have no tool calls "
            "left; answer in plain text, briefly explaining what went wrong."
        )
    return (
        f"Observation: your tool call failed — {error}. Fix the problem (correct the "
        "tool name or arguments against the TOOLS list) and respond with ONLY the "
        "corrected <tool_call>. If it can't be fixed by adjusting the call, answer "
        "in plain text instead."
    )


def repeated_call_blocked(is_last_hop: bool) -> str:
    """Observation injected when the model tries to run the exact same tool
    call (name + arguments) it already ran this turn — repeating it cannot
    produce new information, so block execution instead of burning a hop."""
    if is_last_hop:
        return (
            "Observation: you already called that exact tool with the same arguments "
            "this turn — running it again won't produce new information. You have no "
            "tool calls left; answer now in plain text using what you already have, or "
            "ask the user directly for whatever's missing."
        )
    return (
        "Observation: you already called that exact tool with the same arguments this "
        "turn — running it again won't produce new information. Either try a genuinely "
        "different tool/arguments, ask the user directly for what's missing, or answer "
        "now in plain text."
    )


def force_final_answer() -> str:
    """Used when the hop budget is exhausted: forces a real, non-empty reply
    instead of letting the turn end with nothing shown to the user."""
    return (
        "You're out of tool calls for this turn. Write your final answer now, in plain "
        "text. Use whatever you learned from the observations above; if that's not "
        "enough to fully answer, say what you found so far and what's still missing — "
        "but you must write something."
    )


def foreign_script_correction() -> str:
    """Observation injected when the model hand-wrote non-Latin script that
    didn't come from a tool (its own non-English output is unreliable)."""
    return (
        "Observation: your draft answer contained non-English script that you wrote "
        "yourself. You cannot write non-English scripts reliably — respond with ONLY "
        "a <tool_call> to 'translate' (put the English text in 'text' and the "
        "language the user wants in 'target_language_code'), then relay its output."
    )
