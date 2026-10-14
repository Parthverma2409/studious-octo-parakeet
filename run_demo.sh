#!/usr/bin/env bash
# One-command demo launcher for macOS / Linux:   ./run_demo.sh
# Creates the virtual environment on first run, installs the project,
# generates the NLU data the demo needs, then runs the narrated walkthrough.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found. Install Python 3.11 or newer."
    exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv .venv
else
    echo "[1/4] Virtual environment already exists."
fi
PY=.venv/bin/python

echo "[2/4] Installing project (first run only takes a minute)..."
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -e ".[dev]"

if [ ! -f "data/nlu/splits/train.jsonl" ]; then
    echo "[3/4] Generating NLU dataset splits (needed by demo section 3)..."
    "$PY" -m packages.nlp build >/dev/null
else
    echo "[3/4] NLU dataset splits already present."
fi

echo "[4/4] Running the demo..."
echo
"$PY" -m packages.demo "$@"

echo
echo "---------------------------------------------------------------"
echo "Demo finished. To run one section only:  ./run_demo.sh --only 4"
echo "To list sections:                        ./run_demo.sh --list"
echo "Full setup and other commands:           SETUP.md"
echo "---------------------------------------------------------------"
