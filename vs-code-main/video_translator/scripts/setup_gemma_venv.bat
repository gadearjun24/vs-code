@echo off
REM One-time setup for the SEPARATE Gemma 4 worker venv (.venv-gemma), used
REM only by step11 (transcription + translation). Keeps transformers>=5.10.1
REM (required by Gemma 4) fully isolated from the main venv's
REM transformers<5.0 pin (required by coqui-tts).
REM
REM Usage:
REM   scripts\setup_gemma_venv.bat          CPU build (default, works everywhere)
REM   scripts\setup_gemma_venv.bat cuda     CUDA 12.4 build, if you have an NVIDIA GPU

setlocal
cd /d "%~dp0\.."

set VARIANT=%1
if "%VARIANT%"=="" set VARIANT=cpu

python -m venv .venv-gemma
call .venv-gemma\Scripts\activate.bat

pip install --upgrade pip

if /I "%VARIANT%"=="cuda" (
    echo Installing torch ^(CUDA 12.4 build^)...
    pip install torch --index-url https://download.pytorch.org/whl/cu124
) else (
    echo Installing torch ^(CPU build^)...
    pip install torch --index-url https://download.pytorch.org/whl/cpu
)

pip install -r requirements-gemma.txt

call .venv-gemma\Scripts\deactivate.bat

echo.
echo Gemma worker venv ready at .venv-gemma
echo.
echo Next steps:
echo   1. Accept the license for google/gemma-4-E2B-it (while logged in):
echo      https://huggingface.co/google/gemma-4-E2B-it
echo   2. set HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx   (same token used for pyannote)
echo   3. Run the main pipeline normally - step11 will call this venv automatically.
