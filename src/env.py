"""Loads key=value pairs from a .env file in the project root into
os.environ, without a python-dotenv dependency. Real environment variables
(already set in the shell) always win — .env only fills in what's missing,
so `export GROQ_API_KEY=...` still overrides whatever is in the file."""

import os

_ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")


def load_dotenv() -> None:
    if not os.path.isfile(_ENV_PATH):
        return

    with open(_ENV_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value
