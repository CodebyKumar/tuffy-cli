"""Voice mode CLI loop for Tuffy.

Coordinates microphone input, transcribing, LLM execution, text cleaning,
and text-to-speech feedback.
"""

import sys
import re
from typing import Iterator

from src.cli.session import Session
from src.cli.commands import handle_command
from src.cli.turn import run_turn
from src.cli.display import C_DIM, C_USER, C_RESET, C_AI, C_WARN

from src.voice.audio import AudioInterface
from src.voice.stt import WhisperSTT
from src.voice.tts import PiperTTS


def clean_text_for_speech(text: str) -> str:
    """Strips markdown and formatting so that text-to-speech sounds natural.

    Skips code blocks, removes asterisks, backticks, list indicators, and excessive spaces.
    """
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", " [code block omitted] ", text)
    # Remove inline code backticks
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove bold/italic markdown formatting
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    # Remove markdown header lines
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
    # Remove markdown link syntaxes, keeping the link text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Clean up double spaces or line endings
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _get_last_assistant_reply(session: Session) -> str:
    """Finds and returns the text of the latest assistant reply in history."""
    for msg in reversed(session.messages):
        if msg.get("role") == "assistant":
            content = msg.get("content")
            return content if isinstance(content, str) else ""
    return ""


def start_voice_session(session: Session) -> None:
    """Starts the interactive voice-based conversation loop."""
    print(f"\n{C_DIM}Loading voice modules (Whisper STT & Piper TTS)...{C_RESET}")
    
    stt = None
    try:
        try:
            stt = WhisperSTT()
            tts = PiperTTS()
            audio = AudioInterface()
        except Exception as e:
            print(f"\n{C_WARN}Error initializing voice components: {e}{C_RESET}")
            print("Please check that your dependencies are installed: `uv sync --extra voice`")
            print("Falling back to standard text-only mode.")
            return

        print(f"{C_DIM}Voice components loaded. Ready!{C_RESET}")
        print(f"{C_DIM}Press [Enter] on an empty line to speak, or type commands directly.{C_RESET}\n")

        while True:
            try:
                # Show standard user prompt
                user_input = input(f"{C_USER}You (Enter to record, or type message) ❯{C_RESET} ")
            except (KeyboardInterrupt, EOFError):
                print()
                session.end()
                print(f"{C_DIM}Goodbye!{C_RESET}")
                return "exit"

            stripped = user_input.strip()

            # 1. Handle normal text/command input
            if stripped:
                if stripped.startswith("/"):
                    cmd_lower = stripped.lower()
                    if cmd_lower == "/mode" or cmd_lower.startswith("/mode "):
                        mode = cmd_lower[len("/mode"):].strip()
                        if not mode:
                            print(f"{C_DIM}Current mode: voice. Use '/mode text' to switch.{C_RESET}\n")
                            continue
                        if mode == "text":
                            print(f"{C_DIM}Switching to text mode.{C_RESET}\n")
                            return
                        elif mode == "voice":
                            print(f"{C_DIM}Already in voice mode.{C_RESET}\n")
                            continue
                        else:
                            print(f"{C_DIM}Unknown mode: {mode}. Use '/mode voice' or '/mode text'.{C_RESET}\n")
                            continue

                    result = handle_command(session, stripped)
                    if result == "exit":
                        session.end()
                        print(f"{C_DIM}Goodbye!{C_RESET}")
                        return "exit"
                    if result == "handled":
                        continue
                    print(f"{C_DIM}Unknown command: {stripped}. Type /help for a list.{C_RESET}\n")
                    continue

                # Run text-only turn (no TTS playback, as user typed text)
                run_turn(session, stripped)
                continue

            # 2. Empty Enter: Run voice mode (mic record -> transcribe -> turn -> play audio)
            print(f"{C_DIM}Recording... Press [Enter] to stop.{C_RESET}", end="", flush=True)

            # Capture audio from the microphone. A mic/ALSA failure here (unplugged
            # device, missing arecord, PortAudio device busy) must not crash the
            # whole assistant process - drop back to the voice prompt instead.
            try:
                audio_data = audio.record_audio()
            except Exception as e:
                print(f"\r{C_WARN}Recording failed: {e}                               {C_RESET}\n")
                continue

            # Backspace the "Recording..." prompt line to clean up the screen
            print(f"\r{C_DIM}Transcribing...                         {C_RESET}", end="", flush=True)
            
            try:
                transcribed_text = stt.transcribe(audio_data)
            except Exception as e:
                print(f"\r{C_WARN}Transcription failed: {e}                               {C_RESET}\n")
                continue

            # Clear the transcribing status line
            print("\r" + " " * 50 + "\r", end="", flush=True)

            if not transcribed_text.strip():
                print(f"{C_DIM}[No speech detected — try again]{C_RESET}\n")
                continue

            # Print what the user said
            print(f"{C_USER}You ❯{C_RESET} {transcribed_text}")

            # Run the agent turn (prints thoughts, tool execution, and answers live)
            success = run_turn(session, transcribed_text)
            if not success:
                continue

            # Get response text and speak it back
            reply = _get_last_assistant_reply(session)
            cleaned_reply = clean_text_for_speech(reply)
            if cleaned_reply:
                print(f"{C_DIM}Speaking... (Press [Enter] to stop){C_RESET}", end="", flush=True)
                
                import threading
                import select
                
                stop_event = threading.Event()
                
                def play_target():
                    try:
                        pcm_chunks = tts.synthesize(cleaned_reply)
                        audio.play_audio(pcm_chunks, sample_rate=tts.sample_rate, stop_event=stop_event)
                    except Exception as play_err:
                        print(f"\n{C_WARN}Audio playback error: {play_err}{C_RESET}")

                play_thread = threading.Thread(target=play_target)
                play_thread.start()
                
                interrupted = False
                try:
                    while play_thread.is_alive():
                        # Check for Enter key press on stdin (POSIX select)
                        rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                        if rlist:
                            # Consume the input (newline) so it doesn't feed into next turn
                            sys.stdin.readline()
                            stop_event.set()
                            interrupted = True
                            break
                except Exception:
                    # Fallback if select is not supported in the current terminal environment
                    play_thread.join()
                
                play_thread.join()
                
                # Clear status line
                if interrupted:
                    print(f"\r{C_DIM}Speaking... [Interrupted]                  {C_RESET}")
                else:
                    print("\r" + " " * 45 + "\r", end="", flush=True)
    finally:
        if stt is not None:
            from src.voice.stt import suppress_stdout_stderr
            with suppress_stdout_stderr():
                del stt
