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
Every Assistant turn below starts with a <think> block — this is mandatory on every reply, tool call or not. Every reply that isn't a <tool_call> wraps its actual answer in <final_response>...</final_response> — nothing outside <think> and <final_response> tags, ever.

Example — plain reply, no <tool_call> tag at all:
User: hey, how's it going?
Assistant: <think>Just a greeting.</think>
<final_response>All good — what can I do for you?</final_response>

Example — one tool:
User: <something you need a tool for>
Assistant: <think>brief reasoning about what's needed</think>
<tool_call>
{"thought": "<one short line: what the user needs -> which tool>", "name": "<tool name from the list>", "arguments": {"<arg>": "<value taken from the user's actual message>"}}
</tool_call>
(after the Observation comes back)
Assistant: <think>brief reasoning grounded in the observation</think>
<final_response><final answer grounded in the observation></final_response>

Example — chained tools:
User: <something needing two steps, e.g. look something up then translate it>
Assistant: <think>first I need X</think>
<tool_call>
{"thought": "first I need X", "name": "<tool A>", "arguments": {...}}
</tool_call>
(Observation A comes back)
Assistant: <think>now convert/refine with tool B</think>
<tool_call>
{"thought": "now convert/refine with tool B", "name": "<tool B>", "arguments": {...}}
</tool_call>
(Observation B comes back)
Assistant: <final_response><final answer></final_response>"""


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
    context_plan = None,
    skill_lines: list = None,
) -> str:
    """The dynamic part of the system prompt: tool signatures, long-term
    memory, episodic session summaries, self-learned lessons, and installed
    skills."""
    sections = []

    sections.append("TOOLS YOU CAN CALL\n" + "\n".join(tool_lines))

    if skill_lines:
        sections.append(
            "SKILLS AVAILABLE (call read_skill(name) for full guidance when one "
            "looks relevant to what the user is asking):\n" + "\n".join(skill_lines)
        )

    if context_plan:
        user_facts = context_plan.sections.get("user_facts", "").strip()
        if user_facts:
            sections.append(
                "WHAT YOU KNOW ABOUT THE USER (long-term memory — answer from this "
                "directly, no tool call needed):\n" + user_facts
            )
        else:
            sections.append(
                "WHAT YOU KNOW ABOUT THE USER: nothing stored yet. As you learn "
                "facts about them, they will appear here."
            )

        relevant_moments = context_plan.sections.get("relevant_past_moments", "").strip()
        if relevant_moments:
            sections.append(
                "RELEVANT PAST MOMENTS (from your memory of previous conversations):\n"
                + relevant_moments
            )

        prev_sessions = context_plan.sections.get("previous_sessions", "").strip()
        if prev_sessions:
            sections.append(
                "PREVIOUS SESSIONS:\n" + prev_sessions
            )

        lessons = context_plan.sections.get("lessons", "").strip()
        if lessons:
            sections.append(
                "LESSONS FROM YOUR OWN PAST MISTAKES (apply these):\n" + lessons
            )

        if getattr(context_plan, "rolling_summary", None):
            sections.append(
                "EARLIER IN THIS CONVERSATION (condensed):\n" + context_plan.rolling_summary
            )
    else:
        sections.append(
            "WHAT YOU KNOW ABOUT THE USER: nothing stored yet. As you learn "
            "facts about them, they will appear here."
        )

    sections.append("PROTOCOL EXAMPLES\n" + _REACT_EXAMPLES)

    return "\n\n".join(sections)


def tool_output_prompt(function_name: str, tool_output: str, is_last_hop: bool) -> str:
    """The ReAct Observation message appended after a tool call succeeds."""
    if is_last_hop:
        next_step = (
            "You have no tool calls left. Give your final answer now, wrapped in "
            "<final_response>...</final_response>, grounded in this observation — "
            "respond to what the user actually asked, don't just repeat the observation back."
        )
    else:
        next_step = (
            "Decide: if you still need another step, respond with ONLY the next "
            "<tool_call> (with a fresh thought). Otherwise give your final answer "
            "wrapped in <final_response>...</final_response>, grounded in this "
            "observation — respond to what the user actually asked, don't just repeat "
            "the observation back."
        )
    return f"Observation from {function_name}:\n{tool_output}\n\n{next_step}"


def tool_call_failed(error: str, is_last_hop: bool = True) -> str:
    """Observation message when a <tool_call> fails to parse or execute."""
    if is_last_hop:
        return (
            f"Observation: your tool call failed — {error}. You have no tool calls "
            "left; answer in <final_response>...</final_response>, briefly explaining "
            "what went wrong."
        )
    return (
        f"Observation: your tool call failed — {error}. Fix the problem (correct the "
        "tool name or arguments against the TOOLS list) and respond with ONLY the "
        "corrected <tool_call>. If it can't be fixed by adjusting the call, answer in "
        "<final_response>...</final_response> instead."
    )


def repeated_call_blocked(is_last_hop: bool) -> str:
    """Observation injected when the model tries to run the exact same tool
    call (name + arguments) it already ran this turn — repeating it cannot
    produce new information, so block execution instead of burning a hop."""
    if is_last_hop:
        return (
            "Observation: you already called that exact tool with the same arguments "
            "this turn — running it again won't produce new information. You have no "
            "tool calls left; answer now in <final_response>...</final_response> using "
            "what you already have, or ask the user directly for whatever's missing."
        )
    return (
        "Observation: you already called that exact tool with the same arguments this "
        "turn — running it again won't produce new information. Either try a genuinely "
        "different tool/arguments, ask the user directly for what's missing, or answer "
        "now in <final_response>...</final_response>."
    )


def same_tool_called_again_prefix(function_name: str, previous_output: str) -> str:
    """Prefix prepended to the normal tool_output_prompt Observation when the
    model calls a tool it ALREADY called this turn, with different arguments
    (not blocked outright like an exact repeat — a second, genuinely
    different search is legitimate). The call still runs; this just makes
    sure the model can't "forget" the first result and treat a near-
    duplicate call as if nothing had happened — restates it inline so
    there's no excuse to call the tool a third time just to "double check"."""
    return (
        f"(Note: you already called {function_name} once this turn and got: "
        f"{previous_output}\n\n"
        f"Only call {function_name} again if this new attempt is asking for something "
        "genuinely different — if the result above already answers the user's question, "
        "answer now in <final_response>...</final_response> instead.)\n\n"
    )


def force_final_answer() -> str:
    """Used when the hop budget is exhausted: forces a real, non-empty reply
    instead of letting the turn end with nothing shown to the user."""
    return (
        "You're out of tool calls for this turn. Write your final answer now, wrapped "
        "in <final_response>...</final_response>. Use whatever you learned from the "
        "observations above; if that's not enough to fully answer, say what you found "
        "so far and what's still missing — but you must write something."
    )


def no_tool_needed_nudge(is_last_hop: bool) -> str:
    """Observation injected when a <tool_call> tag named something that isn't
    a real tool (blank, hallucinated, or a stray phrase like "no tool
    needed") - treated as the model meaning to skip the tool entirely rather
    than a failure worth scolding it over, since there's nothing concrete to
    correct: it just needs to answer directly instead."""
    if is_last_hop:
        return (
            "That wasn't a real tool, so no tool call happened. Answer directly now, "
            "wrapped in <final_response>...</final_response>, with no <tool_call> tag."
        )
    return (
        "That wasn't a real tool, so no tool call happened. If you don't need a tool, "
        "just answer directly, wrapped in <final_response>...</final_response>, with no "
        "<tool_call> tag at all."
    )


def degenerate_reply_correction() -> str:
    """Observation injected when the model's draft answer collapsed into a
    bare chat-role word ('user'/'assistant'/'system') instead of real
    content — a known small-model failure mode where the next-token
    distribution drifts onto a role-label token straight after the
    PROTOCOL EXAMPLES block's User:/Assistant: lines. Asks for a genuine
    retry rather than showing the stray token to the user."""
    return (
        "Observation: your draft answer wasn't a real response — it came out empty "
        "or as just a stray word. Write your actual answer now, wrapped in "
        "<final_response>...</final_response>, responding to what the user actually said."
    )


def refusal_without_tool_correction(tool_lines: list) -> str:
    """Observation injected when the model's final text claims it can't do
    something ("I don't have access to...", "could you share the code?")
    on a hop where it never attempted a <tool_call> at all - a known 2B-model
    failure mode where a plausible-sounding refusal is produced instead of
    reaching for a tool that's sitting right there in TOOLS YOU CAN CALL.
    Restates the tool list inline (not just "check TOOLS above") since the
    model already had that context once and skipped it - repeating the exact
    names gives it something concrete to act on instead of refusing again."""
    return (
        "Observation: that was a refusal, not an answer - you said you can't do "
        "something without ever calling a tool for it. Before answering that way, "
        "check whether one of your actual tools does this. Available tools:\n"
        + "\n".join(tool_lines)
        + "\n\nIf one of them applies, respond with ONLY that <tool_call> now. If none "
        "of them actually help (not just that you'd rather not), say so plainly in "
        "<final_response>...</final_response>."
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
