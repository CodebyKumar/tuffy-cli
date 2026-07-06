"""System tools: live host-machine resource stats and process listing. Use
these for questions about THIS machine's hardware/performance, never for
general knowledge (that's research.py's web_search)."""

import json
import psutil

from src.tools.registry import registry


@registry.register(
    name="get_system_stats",
    description="Retrieve the currently running host machine's live real-time hardware resource performance metrics (such as CPU, RAM, or GPU load, disk space, battery status, network IO, uptime, and host OS platform version details). Do NOT call this for general knowledge queries or questions about hardware brands/products.",
    parameters={"metric": {"type": "string", "description": "The target metric to check. Must be one of: 'cpu', 'memory', 'gpu', 'disk', 'battery', 'network', 'uptime', 'os', or 'all'."}},
    required=["metric"],
    group="system",
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
    name="top_processes",
    description="List the top processes on this machine by CPU or memory usage. Use this for questions like "
                 "'what's using my RAM' or 'what's eating my CPU'.",
    parameters={
        "sort_by": {"type": "string", "description": "Either 'cpu' or 'memory' — which metric to rank processes by. Defaults to 'cpu'."}
    },
    required=[],
    group="system",
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
