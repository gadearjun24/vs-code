#!/usr/bin/env bash
# One-time setup for the SEPARATE Gemma 4 worker venv (.venv-gemma), used
# only by step11 (transcription + translation). Keeps transformers>=5.10.1
# (required by Gemma 4) fully isolated from the main venv's
# transformers<5.0 pin (required by coqui-tts).
#
# Usage:
#   bash scripts/setup_gemma_venv.sh          # CPU build (default, works everywhere)
#   bash scripts/setup_gemma_venv.sh cuda      # CUDA 12.4 build, if you have an NVIDIA GPU
set -euo pipefail

cd "$(dirname "$0")/.."

VARIANT="${1:-cpu}"

python3 -m venv .venv-gemma
# shellcheck disable=SC1091
source .venv-gemma/bin/activate

pip install --upgrade pip

if [ "$VARIANT" = "cuda" ]; then
    echo "Installing PyTorch (CUDA 12.4)..."
    pip install \
        torch \
        torchvision \
        torchaudio \
        --index-url https://download.pytorch.org/whl/cu124
else
    echo "Installing PyTorch (CPU)..."
    pip install \
        torch \
        torchvision \
        torchaudio \
        --index-url https://download.pytorch.org/whl/cpu
fi

pip install -r requirements-gemma.txt

deactivate

echo ""
echo "Gemma worker venv ready at .venv-gemma"
echo ""
echo "Next steps:"
echo "  1. Accept the license for google/gemma-4-E2B-it (while logged in):"
echo "     https://huggingface.co/google/gemma-4-E2B-it"
echo "  2. export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx   (same token used for pyannote)"
echo "  3. Run the main pipeline normally - step11 will call this venv automatically."
