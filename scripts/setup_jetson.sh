#!/usr/bin/env bash
# One-shot bootstrap for running Tuffy on a Jetson Orin (Tegra/CUDA, JetPack).
# Meant to be run top-to-bottom on a fresh copy of this repo (e.g. copied over
# via a pendrive rather than git-cloned) — installs uv if missing, syncs
# Python deps, rebuilds llama-cpp-python with CUDA (Jetson has no Metal), and
# launches the app.
#
# Usage:
#   cd /path/to/tuffy
#   bash scripts/setup_jetson.sh
#
# Re-run any time to rebuild after pulling new weights/deps — it's idempotent.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "==> Tuffy Jetson Orin setup"
echo "    Project root: $PROJECT_ROOT"

# --- 1. Sanity-check we're actually on a Jetson (Tegra) board -------------
if [ -f /etc/nv_tegra_release ]; then
    echo "==> Detected Jetson/Tegra platform:"
    cat /etc/nv_tegra_release
else
    echo "WARNING: /etc/nv_tegra_release not found — this doesn't look like a Jetson board."
    echo "         Continuing anyway; CUDA build flags below assume Tegra/JetPack's CUDA install."
fi

# --- 2. Install uv if missing ----------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "==> uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "==> uv version: $(uv --version)"

# --- 3. Python version check -------------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found. Install Python 3.11+ (via your JetPack image's apt, or"
    echo "       'uv python install 3.11') before continuing."
    exit 1
fi
echo "==> python3 version: $(python3 --version)"

# --- 4. Sync base dependencies ------------------------------------------
echo "==> Running uv sync..."
uv sync

# --- 5. Rebuild llama-cpp-python with CUDA support ------------------------
# The default llama-cpp-python wheel (or a Mac-built one carried over from
# another machine's .venv) has no GPU backend for Tegra. Force a source
# rebuild against JetPack's CUDA toolkit so n_gpu_layers=-1 in the model
# cards (src/models/configs/local.py) actually offloads to the GPU instead
# of silently falling back to CPU-only inference.
if [ -d /usr/local/cuda ]; then
    echo "==> CUDA toolkit found at /usr/local/cuda — rebuilding llama-cpp-python with CUDA."
    CMAKE_ARGS="-DGGML_CUDA=on" uv pip install --force-reinstall --no-cache-dir --no-binary llama-cpp-python llama-cpp-python
else
    echo "WARNING: /usr/local/cuda not found. Skipping CUDA rebuild of llama-cpp-python —"
    echo "         local models will run CPU-only (slow). Install JetPack's CUDA toolkit and"
    echo "         re-run this script to enable GPU offload."
fi

# --- 6. .env reminder -------------------------------------------------
if [ ! -f .env ] && [ -f .env.example ]; then
    echo "==> No .env found — copying .env.example. Fill in your API keys before using"
    echo "    any API-provider model (/models <groq-model-id>)."
    cp .env.example .env
elif [ ! -f .env ]; then
    echo "==> No .env found. If you're using an API-provider model (e.g. a *-groq model),"
    echo "    create .env in the repo root with e.g. GROQ_API_KEY=... before switching to it."
fi

# --- 7. Install the `tuffy` shell command --------------------------------
# Same launcher used on other machines: cd into the project, activate its
# venv if present, run main.py. TUFFY_HOME is pinned to wherever this script
# actually found the project (PROJECT_ROOT), so it works regardless of where
# the folder ended up on this box (pendrive copy, ~/tuffy, /opt/tuffy, ...).
RC_FILE=""
if [ -n "${ZSH_VERSION:-}" ] || [ "$(basename "${SHELL:-}")" = "zsh" ]; then
    RC_FILE="$HOME/.zshrc"
elif [ -n "${BASH_VERSION:-}" ] || [ "$(basename "${SHELL:-}")" = "bash" ]; then
    RC_FILE="$HOME/.bashrc"
fi

if [ -n "$RC_FILE" ]; then
    if [ -f "$RC_FILE" ] && grep -q "^tuffy() {" "$RC_FILE" 2>/dev/null; then
        echo "==> 'tuffy' command already present in $RC_FILE — leaving it as-is."
    else
        echo "==> Adding 'tuffy' shell command to $RC_FILE"
        cat >> "$RC_FILE" << EOF

# Tuffy agent launcher (added by scripts/setup_jetson.sh)
export TUFFY_HOME="$PROJECT_ROOT"
tuffy() {
    local project_dir="\${TUFFY_HOME:-$PROJECT_ROOT}"
    cd "\$project_dir" || return
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi
    if command -v python >/dev/null 2>&1; then
        python main.py
    else
        python3 main.py
    fi
}
EOF
        echo "    Run 'source $RC_FILE' (or open a new terminal) to start using 'tuffy'."
    fi
else
    echo "WARNING: couldn't detect bash or zsh — skipping 'tuffy' command setup."
    echo "         Run the app directly with: python3 $PROJECT_ROOT/main.py"
fi

echo "==> Setup complete."
echo "==> Launching Tuffy..."
exec python3 main.py
