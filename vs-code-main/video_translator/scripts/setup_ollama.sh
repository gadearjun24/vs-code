#!/usr/bin/env bash
# One-time setup for step11's transcription/translation/emotion engine:
# installs Ollama (if not already present) and pulls the Gemma 4 model.
# This is NOT a Python venv - Ollama is a standalone local server; the
# main pipeline venv only needs the tiny `ollama` client library, already
# in requirements.txt.
#
# Usage:
#   bash scripts/setup_ollama.sh              # pulls gemma4:e2b (default, ~5GB)
#   bash scripts/setup_ollama.sh gemma4:e4b   # higher quality, more RAM
set -euo pipefail

MODEL="${1:-gemma4:e2b}"

if ! command -v ollama >/dev/null 2>&1; then
    echo "Ollama not found - installing..."
    if [ "$(uname)" = "Darwin" ]; then
        echo "On macOS, download the app from https://ollama.com/download instead of this script,"
        echo "or if you have Homebrew: brew install ollama"
        exit 1
    fi
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama already installed: $(ollama --version)"
fi

# The installer usually sets Ollama up as a background service already.
# If `ollama list` fails, start the server manually in another terminal
# with `ollama serve` and re-run this script.
if ! ollama list >/dev/null 2>&1; then
    echo "Ollama server doesn't seem to be running - starting it in the background..."
    nohup ollama serve > /tmp/ollama_serve.log 2>&1 &
    sleep 3
fi

echo "Pulling $MODEL (this downloads the model weights once)..."
ollama pull "$MODEL"

echo ""
echo "Done. Verify with: ollama list"
echo "If you pulled a model other than gemma4:e2b, pass it to the pipeline with:"
echo "    python run_pipeline.py --input input/video.mp4 --target-lang hi --gemma-model-id $MODEL"
