"""Research/lookup tools: web search, translation, datetime. Anything that
answers a question about the outside world (not the host machine itself)
lives here."""

import requests
from datetime import datetime, timedelta
from ddgs import DDGS

from src.tools.registry import registry

_TRANSLATE_ENDPOINT = "https://translate.googleapis.com/translate_a/single"
_TRANSLATE_TIMEOUT = 8


@registry.register(
    name="web_search",
    description="Search the web for general knowledge, lookup of products/brands (like Apple MacBooks), weather, current events, news, or external facts. Use this for general queries, NOT for querying the local host machine's live resource performance.",
    parameters={
        "query": {"type": "string", "description": "The search terms or question to look up."},
        "mode": {"type": "string", "description": "'news' for recent dated news articles, or 'text' for general web results (the default). Choose 'news' for anything time-sensitive."}
    },
    required=["query"],
    group="research",
)
def web_search(query: str, mode: str = "text") -> str:
    try:
        with DDGS() as ddgs:
            if mode.strip().lower() == "news":
                results = list(ddgs.news(query, max_results=5))
            else:
                results = list(ddgs.text(query, max_results=5))

            if not results:
                return "No search results found."

            summary = []
            for i, res in enumerate(results, 1):
                date = res.get("date")
                date_part = f"\nDate: {date}" if date else ""
                snippet = res.get("body") or res.get("excerpt") or ""
                summary.append(
                    f"[{i}] Source: {res.get('href') or res.get('url')}\n"
                    f"Title: {res.get('title')}{date_part}\n"
                    f"Snippet: {snippet}\n"
                )
            return "\n".join(summary)
    except Exception as e:
        return f"Search failed: {str(e)}"


@registry.register(
    name="get_datetime",
    description="Get the current date/time, or compute a date offset (e.g. 'in 10 days', '3 weeks ago'). "
                 "Use this instead of guessing today's date or doing date math yourself.",
    parameters={
        "offset_days": {"type": "string", "description": "Integer number of days to offset from now; 0 or omitted for the current date/time. Can be negative."}
    },
    required=[],
    group="research",
)
def get_datetime(offset_days: str = "0") -> str:
    try:
        days = int(offset_days) if str(offset_days).strip() else 0
    except ValueError:
        return f"Error: offset_days must be an integer, got '{offset_days}'."

    target = datetime.now() + timedelta(days=days)
    return target.strftime("%A, %Y-%m-%d %H:%M:%S")


@registry.register(
    name="translate",
    description="Translate text into a target language, or detect what language it's written in. Use this "
                "whenever the user wants words in another language: 'translate X to Spanish', 'how do you say X "
                "in French', 'what is 100 called in Kannada', or when they send you non-English text you need to "
                "understand (translate it to 'en'). You cannot write non-English scripts yourself — always go "
                "through this tool.",
    parameters={
        "text": {"type": "string", "description": "The text to translate, or to detect the language of."},
        "target_language_code": {"type": "string", "description": "ISO 639-1 target language code (e.g. 'fr' for French, 'es' for Spanish, 'kn' for Kannada, 'en' for English). Omit this to only detect the language of 'text' instead of translating it."}
    },
    required=["text"],
    group="research",
)
def translate(text: str, target_language_code: str = "") -> str:
    try:
        target_language_code = (target_language_code or "").strip().lower()

        if not target_language_code:
            # Detection-only mode.
            _, detected_code = _google_translate(text, target_code="en")
            return f"language_code: {detected_code}"

        result, _ = _google_translate(text, target_code=target_language_code)
        return result
    except Exception as e:
        return f"Translation failed: {str(e)}"


def _google_translate(text: str, target_code: str, source_code: str = "auto") -> tuple[str, str]:
    """Returns (translated_text, detected_source_code) via Google's free translate_a/single endpoint."""
    params = {"client": "gtx", "sl": source_code, "tl": target_code, "dt": "t", "q": text}
    resp = requests.get(_TRANSLATE_ENDPOINT, params=params, timeout=_TRANSLATE_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    translated = "".join(seg[0] for seg in data[0] if seg[0])
    detected_source = data[2] or "en"
    return translated, detected_source
