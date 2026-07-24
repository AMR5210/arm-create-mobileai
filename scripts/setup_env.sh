#!/usr/bin/env bash
# Creates a Python virtual environment and installs project dependencies.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

# Google Colab runs each `!`-prefixed cell in its own fresh subshell, so a venv
# activated inside this script would not carry over to the next notebook cell
# -- every subsequent `!python ...` would silently fall back to the system
# Python and miss everything installed here. Detect Colab and install directly
# into the system Python instead of creating a venv.
if [ -n "${COLAB_RELEASE_TAG:-}" ] || [ -d "/content" ]; then
  echo "==> Colab environment detected: installing directly into the system Python"
  echo "    (a venv here wouldn't persist across Colab's per-cell shells anyway)"
  pip install --upgrade pip
  pip install -r requirements.txt
else
  echo "==> Creating virtual environment at .venv"
  "$PYTHON_BIN" -m venv .venv

  # shellcheck disable=SC1091
  source .venv/bin/activate

  echo "==> Upgrading pip"
  pip install --upgrade pip

  echo "==> Installing Python dependencies"
  pip install -r requirements.txt
fi

echo "==> Checking for build tools required by llama.cpp"
for tool in git cmake; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "WARNING: '$tool' not found on PATH. Install it before running scripts/build_llama_cpp.sh" >&2
  fi
done

echo
echo "Environment ready. Activate it with: source .venv/bin/activate"
echo "Next steps:"
echo "  1. scripts/build_llama_cpp.sh      # build llama.cpp (native, with KleidiAI on Arm)"
echo "  2. scripts/download_model.py       # download the base model"
echo "  3. scripts/prepare_eval_data.py    # prepare wikitext2 + instruction eval set"
