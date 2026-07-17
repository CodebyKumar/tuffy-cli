#!/usr/bin/env bash
# One-shot bootstrap for running Tuffy on a Jetson Orin (Tegra/CUDA, JetPack).
# First run: installs apt build deps, creates the venv, and source-builds
# llama-cpp-python with CUDA (slow, ~20-30 min). Subsequent runs verify the
# existing CUDA build and dependency lockfile are still good and just launch
# — no apt/cmake/rebuild work unless something actually changed.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

MARKER_FILE=".venv/.tuffy_cuda_ready"
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "3.10")

echo "======================================"
echo " Tuffy Jetson Orin Setup"
echo "======================================"
echo

if [[ -f /etc/nv_tegra_release ]]; then
    cat /etc/nv_tegra_release
else
    echo "ERROR: Jetson environment not detected (/etc/nv_tegra_release missing)."
    exit 1
fi

echo

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "uv version:"
uv --version
echo

# --- Locate the JetPack CUDA toolkit -----------------------------------
# JetPack 6.x ships different CUDA minor versions depending on release
# (12.2 through 12.6+), so don't hardcode one. Prefer the /usr/local/cuda
# symlink (always points at the active toolkit); fall back to the newest
# /usr/local/cuda-* directory that actually has nvcc.
CUDA_HOME=""

if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    CUDA_HOME=/usr/local/cuda
else
    for candidate in $(ls -d /usr/local/cuda-* 2>/dev/null | sort -V -r); do
        if [[ -x "$candidate/bin/nvcc" ]]; then
            CUDA_HOME="$candidate"
            break
        fi
    done
fi

if [[ -z "$CUDA_HOME" ]]; then
    echo "ERROR: CUDA toolkit not found under /usr/local/cuda*."
    echo "       Install it via 'sudo apt install nvidia-cuda-toolkit' or the"
    echo "       JetPack SDK Manager, then re-run this script."
    exit 1
fi

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo "CUDA detected at $CUDA_HOME:"
"$CUDA_HOME/bin/nvcc" --version
echo

# llama-cpp-python must NEVER be touched by plain `uv sync` — the default
# PyPI wheel is CPU-only and would silently clobber the CUDA source build
# we install below, forcing a full rebuild on every subsequent run. Both
# the explicit flag (this script's own `uv sync` calls) and the env var
# (uv reads UV_NO_INSTALL_PACKAGE itself, so it also protects any bare
# `uv sync` run directly in this same process/subshell) are set — belt
# and suspenders, since the RC-file guard below only applies to *future*
# shells, not this one.
SYNC_FLAGS=(--no-install-package llama-cpp-python --no-install-package pywhispercpp --no-install-package onnxruntime --inexact --extra voice)
export UV_NO_INSTALL_PACKAGE="llama-cpp-python,pywhispercpp,onnxruntime"

verify_cuda() {
    .venv/bin/python - <<'PY'
import sys
try:
    import llama_cpp
except ImportError as e:
    print(f"ImportError: {e}", file=sys.stderr)
    sys.exit(1)

gpu_support = False
if hasattr(llama_cpp, "llama_supports_gpu_offload") and llama_cpp.llama_supports_gpu_offload():
    gpu_support = True

try:
    info = llama_cpp.llama_print_system_info()
    if info:
        if isinstance(info, bytes):
            info = info.decode()
        if "CUDA" in info.upper() or "GGML_CUDA" in info.upper():
            gpu_support = True
        print(info)
except Exception as e:
    print(f"Warning: failed to print system info: {e}", file=sys.stderr)

if not gpu_support:
    print("CUDA/GPU backend not detected in llama-cpp-python", file=sys.stderr)
    sys.exit(1)
PY
}

verify_whisper_cuda() {
    .venv/bin/python - <<'PY'
import sys
try:
    import pywhispercpp.model as m
    if 'use_gpu' not in m.ContextParams.__annotations__:
        raise RuntimeError("use_gpu key not found in ContextParams annotations")
    print("pywhispercpp CUDA verification succeeded")
except Exception as e:
    print(f"pywhispercpp CUDA verification failed: {e}", file=sys.stderr)
    sys.exit(1)
PY
}

verify_onnx_gpu() {
    .venv/bin/python - <<'PY'
import sys
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' not in providers:
        raise RuntimeError(f"CUDAExecutionProvider not found in {providers}")
    print("onnxruntime CUDA verification succeeded")
except Exception as e:
    print(f"onnxruntime CUDA verification failed: {e}", file=sys.stderr)
    sys.exit(1)
PY
}

# Fingerprint of only what should invalidate the cached CUDA build: the
# llama-cpp-python version constraint (a bump there means a real rebuild) and
# this script itself (CMAKE flags, architecture, etc). Deliberately NOT the
# whole lockfile - unrelated dependency bumps (elastimem, mcp, ...) must not
# trigger a ~20-30 min llama-cpp-python source rebuild for nothing.
current_fingerprint() {
    grep -A2 '"llama-cpp-python' pyproject.toml 2>/dev/null | cat - "$SCRIPT_DIR/setup_jetson.sh" 2>/dev/null \
        | shasum -a 256 | awk '{print $1}'
}

#
# Fast path: environment already set up and untouched since last success.
#
if [[ -d .venv && -f "$MARKER_FILE" ]]; then
    echo "Existing CUDA-enabled Tuffy environment detected."

    if [[ "$(cat "$MARKER_FILE")" == "$(current_fingerprint)" ]] && verify_cuda >/dev/null 2>&1; then
        echo "CUDA backend verified, dependencies unchanged."

        if [[ -f uv.lock ]]; then
            uv sync --frozen "${SYNC_FLAGS[@]}"
        else
            uv sync "${SYNC_FLAGS[@]}"
        fi

        echo
        echo "Launching Tuffy..."
        echo

        exec .venv/bin/python main.py
    else
        echo "Environment is stale or failed verification. Re-validating/rebuilding."
        echo
    fi
fi

echo "======================================"
echo " Validating system requirements"
echo "======================================"
echo

# Only run apt updates and tool installs if packages are missing
if ! command -v cmake >/dev/null 2>&1 || ! command -v ninja >/dev/null 2>&1 || ! dpkg -s libportaudio2 &>/dev/null || ! command -v aplay >/dev/null 2>&1; then
    sudo apt update
    sudo apt install -y \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        git \
        python3-dev \
        python3-pip \
        python3-setuptools \
        libportaudio2 \
        alsa-utils
else
    echo "Build tools and audio libraries already present. Skipping apt install."
fi

echo
if [[ ! -d .venv ]]; then
    echo "Creating virtual environment (Python $PYTHON_VERSION)..."
    uv venv --python "$PYTHON_VERSION"
else
    echo "Virtual environment already exists. Skipping creation."
fi

source .venv/bin/activate

echo
echo "======================================"
echo " Synchronizing project dependencies"
echo "======================================"

if [[ -f uv.lock ]]; then
    uv sync --frozen "${SYNC_FLAGS[@]}"
else
    uv sync "${SYNC_FLAGS[@]}"
fi

echo
echo "======================================"
echo " Validating llama-cpp-python CUDA build"
echo "======================================"

NEED_REBUILD=true

if .venv/bin/python -c "import llama_cpp" >/dev/null 2>&1 && verify_cuda >/dev/null 2>&1; then
    NEED_REBUILD=false
fi

if [[ "$NEED_REBUILD" == "true" ]]; then
    echo "llama-cpp-python is missing or lacks CUDA support. Building from source..."
    echo "(This step compiles llama.cpp for Jetson Orin's SM 8.7 GPU and takes a while.)"
    echo

    export CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87"
    export FORCE_CMAKE=1

    uv pip uninstall -y llama-cpp-python || true

    uv pip install \
        --force-reinstall \
        --no-cache-dir \
        --no-binary llama-cpp-python \
        "llama-cpp-python>=0.3.32"
else
    echo "llama-cpp-python is already compiled with CUDA. Skipping rebuild."
fi

echo
echo "======================================"
echo " Validating pywhispercpp CUDA build"
echo "======================================"

NEED_WHISPER_REBUILD=true

if .venv/bin/python -c "import pywhispercpp" >/dev/null 2>&1 && verify_whisper_cuda >/dev/null 2>&1; then
    NEED_WHISPER_REBUILD=false
fi

if [[ "$NEED_WHISPER_REBUILD" == "true" ]]; then
    echo "pywhispercpp is missing or lacks CUDA support. Building from source..."
    echo "(This compiles whisper.cpp for Jetson Orin's SM 8.7 GPU.)"
    echo

    export CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87"
    export GGML_CUDA=1
    export WHISPER_CUDA=1

    uv pip uninstall -y pywhispercpp || true

    uv pip install \
        --force-reinstall \
        --no-cache-dir \
        --no-binary pywhispercpp \
        "pywhispercpp>=1.5.0"
else
    echo "pywhispercpp is already compiled with CUDA. Skipping rebuild."
fi

echo
echo "======================================"
echo " Validating onnxruntime-gpu for Jetson"
echo "======================================"

# Parse JetPack and CUDA version for the Jetson AI Lab wheels index
L4T_RELEASE=$(head -n 1 /etc/nv_tegra_release | grep -o -E "R[0-9]+") || true
if [[ "$L4T_RELEASE" == "R36" ]]; then
    JP_VERSION="jp6"
elif [[ "$L4T_RELEASE" == "R35" ]]; then
    JP_VERSION="jp5"
else
    JP_VERSION="jp6"
fi

CUDA_VER=$("$CUDA_HOME/bin/nvcc" --version | grep -i -o -E "release [0-9]+\.[0-9]+" | cut -d' ' -f2) || true
if [[ -n "$CUDA_VER" ]]; then
    CUDA_SUFFIX="cu$(echo "$CUDA_VER" | tr -d '.')"
else
    CUDA_SUFFIX="cu122"
fi

JETSON_PIP_INDEX="https://pypi.jetson-ai-lab.io/$JP_VERSION/$CUDA_SUFFIX"

NEED_ONNX_GPU=true
if verify_onnx_gpu >/dev/null 2>&1; then
    NEED_ONNX_GPU=false
fi

if [[ "$NEED_ONNX_GPU" == "true" ]]; then
    echo "Installing GPU-enabled onnxruntime for Jetson from $JETSON_PIP_INDEX..."
    echo
    
    uv pip uninstall -y onnxruntime || true
    
    uv pip install \
        --extra-index-url "$JETSON_PIP_INDEX" \
        --no-cache-dir \
        "onnxruntime-gpu"
else
    echo "onnxruntime-gpu is already installed and verified with CUDA support."
fi

echo
echo "======================================"
echo " Final validation"
echo "======================================"

verify_cuda
verify_whisper_cuda
verify_onnx_gpu

current_fingerprint > "$MARKER_FILE"

echo
echo "======================================"
echo " Writing launcher helper"
echo "======================================"

# $SHELL is the user's *login* shell (from /etc/passwd), which doesn't
# always match the shell actually running this terminal (e.g. login shell
# is zsh but the terminal launched bash). Write to every rc file that
# could plausibly be sourced — bash and zsh both, whichever exist — so
# `tuffy` works regardless of which one this session turns out to be.
RC_FILES=()
[[ -f "$HOME/.bashrc" ]] && RC_FILES+=("$HOME/.bashrc")
[[ -f "$HOME/.zshrc" ]] && RC_FILES+=("$HOME/.zshrc")

# Neither exists yet (fresh account) — create the one matching the login
# shell so there's at least one place the block lands.
if [[ ${#RC_FILES[@]} -eq 0 ]]; then
    case "$(basename "${SHELL:-}")" in
        zsh)  RC_FILES=("$HOME/.zshrc") ;;
        *)    RC_FILES=("$HOME/.bashrc") ;;
    esac
fi

for RC_FILE in "${RC_FILES[@]}"; do
    # Strip any previously written block (marked by these sentinels) so
    # re-running the script always refreshes tuffy()/uv() instead of
    # silently keeping a stale version forever.
    if [[ -f "$RC_FILE" ]] && grep -q '# >>> tuffy launcher >>>' "$RC_FILE"; then
        sed -i.bak '/# >>> tuffy launcher >>>/,/# <<< tuffy launcher <<</d' "$RC_FILE"
    fi

    cat >> "$RC_FILE" <<EOF

# >>> tuffy launcher >>>
export TUFFY_HOME="$PROJECT_ROOT"

tuffy() {
    cd "\$TUFFY_HOME" || return
    source .venv/bin/activate
    python main.py
}

# "uv sync" must never reinstall llama-cpp-python's PyPI wheel over the
# CUDA source build. UV_NO_INSTALL_PACKAGE is uv's own env var for this
# (see "uv sync --help") — set globally rather than scoped to \$TUFFY_HOME
# so it also covers a bare "uv sync" run from a script, cron job, or any
# non-interactive shell that sourced this rc file without going through
# the uv() function below. Harmless for any other project, since it only
# ever affects a package actually named llama-cpp-python.
export UV_NO_INSTALL_PACKAGE="llama-cpp-python,pywhispercpp,onnxruntime"

uv() {
    if [[ "\$PWD" == "\$TUFFY_HOME"* && "\$1" == "sync" ]]; then
        command uv sync --no-install-package llama-cpp-python --no-install-package pywhispercpp --no-install-package onnxruntime --inexact --extra voice "\${@:2}"
    else
        command uv "\$@"
    fi
}
# <<< tuffy launcher <<<
EOF
done

echo
echo "======================================"
echo " Setup completed successfully"
echo "======================================"

echo
echo "Launching Tuffy..."
echo

exec .venv/bin/python main.py
