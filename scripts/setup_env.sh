#!/usr/bin/env bash
# Creates a Python virtual environment and installs project dependencies.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "==> Creating virtual environment at .venv"
"$PYTHON_BIN" -m venv .venv

# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip"
pip install --upgrade pip

echo "==> Installing Python dependencies"
pip install -r requirements.txt

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
