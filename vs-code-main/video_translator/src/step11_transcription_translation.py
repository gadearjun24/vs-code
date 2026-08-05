"""
Step 11 - Transcription + translation, combined, using Gemma 4 E2B.

Replaces the old two-step faster-whisper (transcribe) + NLLB-200
(translate) pair with a single audio-native model call per segment:
Gemma 4 E2B listens to the segment's own audio clip and returns BOTH the
original-language transcription and the target-language translation in
one pass, additionally conditioned on:

  - the speaker's gender (from step10's speaker_analysis.json), so the
    translation uses grammatically-correct, natural-sounding pronouns/verb
    agreement for that speaker in languages that mark gender, and
  - the segment's own spoken duration (from diarization), so the
    translation is phrased to fit naturally in roughly that many seconds
    when the dub voice actor (step13's XTTS/MMS clone) speaks it, instead
    of translating in a timing vacuum.

Why this runs AFTER speaker aggregation/normalization/embedding
-----------------------------------------------------------------
Gender is only known once step10 (speaker embedding/gender/age/emotion)
has run - and step10 itself depends on step08 (aggregation) and step09
(normalization). So this step now runs *after* those three (they were
step09/step10/step11 in the old numbering; they're step08/step09/step10
now) instead of before them, which is why the old "step 8 transcription"
slot moved to step11. Nothing about step08/09/10's own logic changed -
only their position in the pipeline.

Why this runs as a subprocess in a separate venv
---------------------------------------------------
Gemma 4 requires `transformers>=5.10.1`. The rest of this pipeline (step13
voice cloning via coqui-tts) requires `transformers<5.0`. Those two
requirements cannot be satisfied in one interpreter/venv, so this step
shells out to a second, independent virtual environment
(`.venv-gemma`, built from `requirements-gemma.txt` -
see `scripts/setup_gemma_venv.sh`) and runs
`gemma_worker/transcribe_translate_worker.py` there. Only plain JSON
crosses the process boundary - no shared Python objects, no import
conflicts.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from config import PipelineConfig, language_name
from src.utils import get_logger, load_json, save_json

LOG = get_logger(__name__)

WORKER_SCRIPT = Path(__file__).resolve().parent.parent / "gemma_worker" / "transcribe_translate_worker.py"


def _check_gemma_venv(cfg: PipelineConfig) -> None:
    gemma_python = cfg.gemma_python
    if gemma_python.exists():
        return
    raise FileNotFoundError(
        f"Gemma worker venv not found at '{cfg.gemma_venv_dir}' (expected python at "
        f"'{gemma_python}').\n"
        f"Set it up once with:\n"
        f"    bash scripts/setup_gemma_venv.sh        # Linux/macOS\n"
        f"    scripts\\setup_gemma_venv.bat            # Windows\n"
        f"(see README.md section 1b - 'Second venv for Gemma 4')."
    )


def _load_gender_map(cfg: PipelineConfig) -> dict:
    if not cfg.speaker_analysis_json.exists():
        LOG.warning(
            f"{cfg.speaker_analysis_json} not found - proceeding without per-speaker gender "
            f"context (translations will use neutral phrasing). Run step10 first for "
            f"gender-aware translation."
        )
        return {}
    analysis = load_json(cfg.speaker_analysis_json)
    return {row["speaker_id"]: row.get("gender") for row in analysis}


def _build_jobs(cfg: PipelineConfig) -> list:
    diarization = load_json(cfg.diarization_json)
    gender_map = _load_gender_map(cfg)

    source_lang_code: Optional[str] = None if cfg.source_lang == "auto" else cfg.source_lang
    source_lang_name = language_name(source_lang_code) if source_lang_code else None
    target_lang_code = cfg.target_lang
    target_lang_name = language_name(target_lang_code)

    jobs = []
    skipped = 0

    for row in diarization:
        speaker_id, segment_id = row["speaker_id"], row["segment_id"]
        audio_file = cfg.dir_speakers / speaker_id / f"{segment_id}.wav"

        if not audio_file.exists():
            LOG.warning(f"  audio not found for {segment_id}, skipping")
            skipped += 1
            continue

        jobs.append({
            "segment_id": segment_id,
            "speaker_id": speaker_id,
            "audio_file": str(audio_file),
            "start": row["start"],
            "end": row["end"],
            "duration": row["duration"],
            "gender": gender_map.get(speaker_id),
            "source_lang_code": source_lang_code,
            "source_lang_name": source_lang_name,
            "target_lang_code": target_lang_code,
            "target_lang_name": target_lang_name,
        })

    if skipped:
        LOG.warning(f"Skipped {skipped} segment(s) with missing audio.")

    return jobs


def _run_worker(cfg: PipelineConfig, jobs_path: Path, results_path: Path) -> None:
    cmd = [
        str(cfg.gemma_python),
        str(WORKER_SCRIPT),
        "--jobs", str(jobs_path),
        "--output", str(results_path),
        "--model-id", cfg.gemma_model_id,
        "--device", cfg.gemma_device,
        "--max-new-tokens", str(cfg.gemma_max_new_tokens),
        "--max-audio-seconds", str(cfg.gemma_max_audio_seconds),
    ]

    LOG.info(f"Launching Gemma worker: {' '.join(cmd)}")

    env = None
    if cfg.hf_token:
        import os
        env = os.environ.copy()
        env["HF_TOKEN"] = cfg.hf_token

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    start = time.time()
    stdout_lines = []

    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip("\n")
        stdout_lines.append(line)

        # Worker emits either "[gemma-worker] ..." log lines or single-line
        # JSON progress records - surface both through our own logger.
        parsed_progress = None
        if line.startswith("{"):
            try:
                obj = json.loads(line)
                if "progress" in obj:
                    parsed_progress = obj
            except json.JSONDecodeError:
                pass

        if parsed_progress:
            LOG.info(
                f"  [{parsed_progress['progress']}/{parsed_progress['total']}] "
                f"{parsed_progress['segment_id']}"
            )
        elif line:
            LOG.info(f"  {line}")

        if time.time() - start > cfg.gemma_worker_timeout_seconds:
            process.kill()
            raise TimeoutError(
                f"Gemma worker exceeded gemma_worker_timeout_seconds="
                f"{cfg.gemma_worker_timeout_seconds}s and was killed."
            )

    returncode = process.wait()
    if returncode != 0:
        tail = "\n".join(stdout_lines[-40:])
        raise RuntimeError(f"Gemma worker exited with code {returncode}. Last output:\n{tail}")


def run(cfg: PipelineConfig) -> list:
    if not cfg.diarization_json.exists():
        raise FileNotFoundError(cfg.diarization_json)

    _check_gemma_venv(cfg)

    cfg.dir_conversation.mkdir(parents=True, exist_ok=True)
    cfg.dir_translations.mkdir(parents=True, exist_ok=True)

    jobs = _build_jobs(cfg)
    total = len(jobs)
    LOG.info(f"{total} segment(s) queued for Gemma 4 transcription + translation.")

    if total == 0:
        save_json(cfg.conversation_json, [])
        save_json(cfg.translated_json, [])
        return []

    jobs_path = cfg.dir_translations / "gemma_jobs.json"
    results_path = cfg.dir_translations / "gemma_results.json"
    save_json(jobs_path, jobs)

    start = time.time()
    _run_worker(cfg, jobs_path, results_path)

    results_by_segment = {r["segment_id"]: r for r in load_json(results_path)}

    conversation = []
    success = failed = 0

    for job in jobs:
        segment_id = job["segment_id"]
        result = results_by_segment.get(segment_id)

        if result is None:
            LOG.error(f"  {segment_id}: no result returned by worker")
            failed += 1
            continue

        if not result.get("ok", False):
            LOG.warning(f"  {segment_id}: {result.get('error')}")

        entry = {
            "segment_id": segment_id,
            "speaker_id": job["speaker_id"],
            "audio_file": job["audio_file"],
            "start": job["start"],
            "end": job["end"],
            "duration": job["duration"],
            "language": result.get("detected_language") or job.get("source_lang_code"),
            "language_probability": None,  # Gemma 4 doesn't expose a confidence score
            "text": result.get("original_text", ""),
            "translation": result.get("translation", ""),
            "emotion": None,
            "gender": job.get("gender"),
            "age_group": None,
            "confidence": None,
            "words": [],  # word-level timestamps are a faster-whisper-only feature; unused downstream
            "asr_translation_engine": cfg.gemma_model_id,
            "translation_ok": bool(result.get("ok", False)),
        }
        conversation.append(entry)

        if result.get("ok", False):
            success += 1
        else:
            failed += 1

    save_json(cfg.conversation_json, conversation)
    save_json(cfg.translated_json, conversation)

    LOG.info(
        f"Transcription + translation done in {time.time() - start:.2f}s | "
        f"success={success} failed={failed}"
    )
    return conversation


if __name__ == "__main__":
    run(PipelineConfig())
