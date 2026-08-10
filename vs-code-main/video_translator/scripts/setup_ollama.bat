@echo off
REM One-time setup for step11's transcription/translation/emotion engine:
REM installs Ollama (if not already present) and pulls the Gemma 4 model.
REM This is NOT a Python venv - Ollama is a standalone local server; the
REM main pipeline venv only needs the tiny `ollama` client library, already
REM in requirements.txt.
REM
REM Usage:
REM   scripts\setup_ollama.bat              pulls gemma4:e2b (default, ~5GB)
REM   scripts\setup_ollama.bat gemma4:e4b   higher quality, more RAM

setlocal
set MODEL=%1
if "%MODEL%"=="" set MODEL=gemma4:e2b

where ollama >nul 2>nul
if errorlevel 1 (
    echo Ollama not found. Download and run the installer from:
    echo     https://ollama.com/download
    echo Then re-run this script.
    exit /b 1
) else (
    echo Ollama already installed.
)

echo Pulling %MODEL% (this downloads the model weights once)...
ollama pull %MODEL%

echo.
echo Done. Verify with: ollama list
echo If you pulled a model other than gemma4:e2b, pass it to the pipeline with:
echo     python run_pipeline.py --input input\video.mp4 --target-lang hi --gemma-model-id %MODEL%
