"""Runs one user turn to completion: builds the message list, streams the
agent's reply token by token through the spinner, and folds the result back
into session history."""

import src.memory as memory
from src.cli.session import Session, keep_only_latest_image, trim_history, compact_turn
from src.cli.display import Spinner, C_DIM, C_RESET
from src.llm.base import ProviderError


def run_turn(session: Session, user_input: str) -> bool:
    """Runs one user turn to completion. Returns False if generation was
    interrupted/failed and the turn was rolled back, True otherwise."""
    memory.mem.tick()
    plan = memory.mem.build_context(user_input)

    messages = session.messages
    messages[0] = session.system_message(context_plan=plan)
    user_message = {"role": "user", "content": user_input}
    if session.pending_image_data_uri:
        user_message = session.agent.attach_image(user_message, session.pending_image_data_uri)
        session.pending_image_data_uri = None
    messages.append(user_message)
    keep_only_latest_image(messages)
    session.messages = messages = trim_history(messages, plan)
    turn_start = len(messages) - 1

    session.trace_printed = False
    print()  # Line gap between "You ❯" and thoughts/spinner

    spinner = Spinner()
    session.active_spinner = spinner
    spinner.start()

    full_response = ""
    try:
        with memory.mem.foreground():
            token_stream = session.agent.run_stream(messages)
            first_token = True
            for token in token_stream:
                # First stop() erases the status text and leaves 'AI ❯ ' in
                # place; later calls are no-ops and never touch the screen.
                if first_token:
                    first_token = False
                    spinner.stop(show_prompt=not session.trace_printed)
                else:
                    spinner.stop(show_prompt=False)
                print(token, end="", flush=True)
                full_response += token
    except RuntimeError as e:
        # llama.cpp returns decode errors (e.g. -3) under memory pressure;
        # drop this turn instead of crashing the whole session.
        spinner.stop()
        memory.mem.report_pressure()
        print(
            f"{C_DIM}[Generation failed: {e}. The machine is likely low on "
            f"memory — close some apps and try again.]{C_RESET}\n"
        )
        del messages[turn_start:]
        return False
    except ProviderError as e:
        # API-model failure (rate limit, bad key, network error, ...); drop
        # this turn instead of crashing the whole session.
        spinner.stop()
        print(f"{C_DIM}[Generation failed: {e}]{C_RESET}\n")
        del messages[turn_start:]
        return False
    except KeyboardInterrupt:
        spinner.stop()
        print(f"{C_DIM}[Interrupted]{C_RESET}\n")
        del messages[turn_start:]
        return False
    finally:
        spinner.stop()
        session.active_spinner = None
    print("\n")

    messages.append({"role": "assistant", "content": full_response})
    # Drop this turn's ReAct intermediates (tool drafts/observations) — the
    # final answer carries what mattered, and stale tool dumps slow every
    # later turn. Ray mode already showed them live; nothing is lost.
    compact_turn(messages, turn_start)

    memory.mem.record_turn(user_input, full_response)
    return True
