"""Tool definitions exposed to the model, plus the private helper logic each
one needs. Memory (memory.py) and workspace file I/O (workspace.py) live in
their own modules; registry.py holds only the ToolRegistry/decorator
machinery. Everything else — search, system stats, translation, datetime —
lives here, one tool per registration, with no tool-specific keyword
sniffing or heuristics: arguments are passed straight through to the
underlying API/library and the model is trusted to supply correct input.

To add a new tool: write a plain function, decorate it with
@registry.register(...), done — no other file needs to change.
"""

import json
import psutil
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
    required=["query"]
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
    name="get_system_stats",
    description="Retrieve the currently running host machine's live real-time hardware resource performance metrics (such as CPU, RAM, or GPU load, disk space, battery status, network IO, uptime, and host OS platform version details). Do NOT call this for general knowledge queries or questions about hardware brands/products.",
    parameters={"metric": {"type": "string", "description": "The target metric to check. Must be one of: 'cpu', 'memory', 'gpu', 'disk', 'battery', 'network', 'uptime', 'os', or 'all'."}},
    required=["metric"]
)
def get_system_stats(metric: str) -> str:
    try:
        metric_lower = metric.lower()
        stats = {}

        if metric_lower in ["cpu", "all"]:
            stats["cpu_percentage"] = f"{psutil.cpu_percent(interval=0.1)}%"
        if metric_lower in ["memory", "all"]:
            mem = psutil.virtual_memory()
            stats["memory_used_gb"] = f"{mem.used / (1024**3):.2f} GB"
            stats["memory_total_gb"] = f"{mem.total / (1024**3):.2f} GB"
            stats["memory_percentage"] = f"{mem.percent}%"
        if metric_lower in ["gpu", "all"]:
            try:
                import subprocess
                out = subprocess.check_output(["system_profiler", "SPDisplaysDataType"]).decode("utf-8")
                gpu_lines = []
                for line in out.splitlines():
                    if any(x in line for x in ["Chipset Model", "Type", "Total Number of Cores", "Vendor", "Metal Support", "VRAM"]):
                        gpu_lines.append(line.strip())
                stats["gpu_info"] = gpu_lines if gpu_lines else "Apple Silicon GPU (native)"
            except Exception as e:
                stats["gpu_info"] = f"Failed to get GPU stats: {e}"
        if metric_lower in ["disk", "all"]:
            disk = psutil.disk_usage('/')
            stats["disk_used_gb"] = f"{disk.used / (1024**3):.2f} GB"
            stats["disk_total_gb"] = f"{disk.total / (1024**3):.2f} GB"
            stats["disk_percentage"] = f"{disk.percent}%"
        if metric_lower in ["battery", "all"]:
            if hasattr(psutil, "sensors_battery"):
                bat = psutil.sensors_battery()
                if bat:
                    stats["battery_percentage"] = f"{bat.percent}%"
                    stats["battery_power_plugged"] = bat.power_plugged
                else:
                    stats["battery_info"] = "No battery detected (e.g. desktop Mac)"
        if metric_lower in ["network", "all"]:
            net = psutil.net_io_counters()
            stats["network_bytes_sent"] = f"{net.bytes_sent / (1024**2):.2f} MB"
            stats["network_bytes_recv"] = f"{net.bytes_recv / (1024**2):.2f} MB"
        if metric_lower in ["uptime", "all"]:
            import time
            boot_time = psutil.boot_time()
            uptime_seconds = time.time() - boot_time
            uptime_hours = uptime_seconds / 3600
            stats["uptime_hours"] = f"{uptime_hours:.2f} hours"
        if metric_lower in ["os", "system", "all"]:
            import platform
            stats["os_platform"] = platform.system()
            stats["os_release"] = platform.release()
            stats["os_version"] = platform.version()
            stats["os_architecture"] = platform.machine()

        return json.dumps(stats, indent=2) if stats else "Error: Invalid metric requested. Choose 'cpu', 'memory', 'gpu', 'disk', 'battery', 'network', 'uptime', 'os' or 'all'."
    except Exception as e:
        return f"Failed to gather system metrics: {str(e)}"


@registry.register(
    name="get_datetime",
    description="Get the current date/time, or compute a date offset (e.g. 'in 10 days', '3 weeks ago'). "
                 "Use this instead of guessing today's date or doing date math yourself.",
    parameters={
        "offset_days": {"type": "string", "description": "Integer number of days to offset from now; 0 or omitted for the current date/time. Can be negative."}
    },
    required=[]
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
    required=["text"]
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


@registry.register(
    name="top_processes",
    description="List the top processes on this machine by CPU or memory usage. Use this for questions like "
                 "'what's using my RAM' or 'what's eating my CPU'.",
    parameters={
        "sort_by": {"type": "string", "description": "Either 'cpu' or 'memory' — which metric to rank processes by. Defaults to 'cpu'."}
    },
    required=[]
)
def top_processes(sort_by: str = "cpu") -> str:
    try:
        sort_key = "memory_percent" if sort_by.strip().lower() == "memory" else "cpu_percent"

        procs = []
        for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                procs.append(p.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        procs.sort(key=lambda p: p.get(sort_key) or 0, reverse=True)
        top = procs[:8]

        lines = [f"PID {p['pid']:>6} | CPU {p.get('cpu_percent', 0):>5.1f}% | MEM {p.get('memory_percent', 0):>5.1f}% | {p.get('name')}" for p in top]
        return "\n".join(lines) if lines else "No process data available."
    except Exception as e:
        return f"Failed to list processes: {str(e)}"
