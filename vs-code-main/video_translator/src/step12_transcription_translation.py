"""
Step 12 - Transcription + translation, combined, using Gemma 4 E2B via Ollama.

Replaces the old two-step faster-whisper (transcribe) + NLLB-200
(translate) pair with a single audio-native model call per segment:
Gemma 4 E2B listens to the segment's own audio clip and returns the
original-language transcription, the target-language translation, and a
detected `emotion` label for that speaker's delivery, all in one pass -
additionally conditioned on:

  - the speaker's gender (from step11's speaker_analysis.json), so the
    translation uses grammatically-correct, natural-sounding pronouns/verb
    agreement for that speaker in languages that mark gender, and
  - the segment's own spoken duration (from diarization), so the
    translation is phrased to fit naturally in roughly that many seconds
    when the dub voice actor (step14's XTTS/MMS clone) speaks it, instead
    of translating in a timing vacuum.

The detected `emotion` is passed through to step14, which uses it to
nudge synthesis delivery (a modest pace/temperature adjustment - XTTS-v2
has no direct "emotion" control - see step14's docstring).

Song vs. dialogue (`is_song` / `contains_dialogue` / `dialogue_windows`)
-----------------------------------------------------------------------
Gemma also flags whether a segment's audio is sung (a song) as opposed to
normally spoken, and - for clips that mix both - roughly WHERE the spoken
dialogue falls (as a fraction of the clip, converted here into absolute
seconds within the clip: `dialogue_windows`). Song lyrics are never
transcribed/translated. step14 uses these flags to decide, per segment,
whether to generate a dub at all or keep the original recording - see its
docstring for the full song/dialogue handling.

Original volume (`original_volume_dbfs`)
-----------------------------------------
Each segment's own original RMS loudness (in dBFS) is measured directly
from its audio here (`_measure_original_volume_dbfs` - a signal
measurement, not something asked of Gemma) and carried through so step14
can make a shouted/angry line come out louder in the dub than a
whispered/calm one, instead of flattening every line's output volume to
the same level.

Why this reads merged_diarization.json instead of diarization.json
-----------------------------------------------------------------
step07 (segment merge) runs right after diarization and merges tiny
same-speaker fragments into natural-length blocks before step08 ever
extracts per-segment audio - so every job built here already corresponds
to a block with enough audio for Gemma 4 to transcribe and read emotion
from reliably, instead of possibly-sub-second raw diarization fragments.
See step07's docstring for the full merge algorithm.

Why this runs AFTER speaker aggregation/normalization/embedding
-----------------------------------------------------------------
Gender is only known once step11 (speaker embedding/gender/age/emotion)
has run - and step11 itself depends on step09 (aggregation) and step10
(normalization). So this step runs after those three rather than before,
which is why the old "step 8 transcription" slot moved to step12.

Why this talks to Ollama instead of importing transformers
---------------------------------------------------------------
Gemma 4 is served locally through Ollama (a standalone server process you
install once - see README section 1b) rather than loaded via
`transformers` in this process. That sidesteps a real dependency
conflict this project used to work around with a second venv: Gemma 4
needed transformers>=5.10.1, while step14's coqui-tts needs
transformers<5.0. Ollama's client library is a tiny HTTP wrapper with no
such constraint, so it lives in the same venv as everything else - see
`gemma_worker/ollama_engine.py` for the actual client logic.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from config import PipelineConfig, language_name
from gemma_worker.ollama_engine import OllamaGemmaEngine
from src.utils import free_memory, get_logger, load_json, rms_dbfs, save_json

LOG = get_logger(__name__)


def _measure_original_volume_dbfs(audio_file: str) -> Optional[float]:
    """
    RMS loudness (dBFS) of the segment's own original audio - measured
    directly from the waveform, not asked of Gemma (loudness is an
    objective signal property, not something worth spending a model call
    guessing at). Stored per segment so step14 can preserve the original
    relative dynamics (a shouted line stays louder than a whispered one in
    the dub) instead of flattening every line to the same output level.
    Returns None if the file can't be read - callers treat that as "no
    volume data available" and fall back to flat normalization for that
    segment only.
    """
    try:
        import soundfile as sf
        audio, _ = sf.read(audio_file, dtype="float32", always_2d=False)
        return round(rms_dbfs(audio), 2)
    except Exception as e:
        LOG.warning(f"  could not measure original volume for {audio_file}: {e}")
        return None


def _dialogue_windows_seconds(dialogue_segments: list, duration: float) -> list:
    """Convert Gemma's 0.0-1.0 clip fractions into absolute seconds within the segment."""
    windows = []
    for seg in dialogue_segments or []:
        start = round(seg["start_fraction"] * duration, 4)
        end = round(seg["end_fraction"] * duration, 4)
        if end > start:
            windows.append({"start": start, "end": end})
    return windows


def _load_gender_map(cfg: PipelineConfig) -> dict:
    if not cfg.speaker_analysis_json.exists():
        LOG.warning(
            f"{cfg.speaker_analysis_json} not found - proceeding without per-speaker gender "
            f"context (translations will use neutral phrasing). Run step11 first for "
            f"gender-aware translation."
        )
        return {}
    analysis = load_json(cfg.speaker_analysis_json)
    return {row["speaker_id"]: row.get("gender") for row in analysis}


def _build_jobs(cfg: PipelineConfig) -> list:
    if not cfg.merged_diarization_json.exists():
        raise FileNotFoundError(
            f"{cfg.merged_diarization_json} not found - run step07 (segment merge) first."
        )
    diarization = load_json(cfg.merged_diarization_json)
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
            # These are the block's TRUE speech start/end/duration (never
            # the possibly-padded clip_start/clip_end step07 may have used
            # just to extract more audio for Gemma) - this is what actually
            # matters for placement/sync and for the "how long does this
            # line have to fit when spoken aloud" budget in the prompt.
            "start": row["start"],
            "end": row["end"],
            "duration": row["duration"],
            "gender": gender_map.get(speaker_id),
            "source_lang_code": source_lang_code,
            "source_lang_name": source_lang_name,
            "target_lang_code": target_lang_code,
            "target_lang_name": target_lang_name,
            # Traceability back to the raw diarization fragments step07
            # merged into this block - not used by the prompt itself, just
            # carried through into translated.json for anyone who wants it.
            "merged_from_count": row.get("merged_from_count", 1),
            "original_segments": row.get("original_segments"),
        })

    if skipped:
        LOG.warning(f"Skipped {skipped} segment(s) with missing audio.")

    return jobs


def run(cfg: PipelineConfig) -> list:
    if not cfg.merged_diarization_json.exists():
        raise FileNotFoundError(
            f"{cfg.merged_diarization_json} not found - run step07 (segment merge) first."
        )

    cfg.dir_conversation.mkdir(parents=True, exist_ok=True)
    cfg.dir_translations.mkdir(parents=True, exist_ok=True)

    log_path = cfg.dir_translations / "ollama_gemma.log"
    engine = OllamaGemmaEngine(
        model_id=cfg.gemma_model_id,
        host=cfg.ollama_host,
        request_timeout=cfg.ollama_request_timeout_seconds,
        num_predict=cfg.ollama_num_predict,
        log_path=log_path,
    )

    LOG.info(f"Checking Ollama at {cfg.ollama_host} for model '{cfg.gemma_model_id}'...")
    engine.check_available()
    LOG.info("Ollama reachable and model present.")

    jobs = _build_jobs(cfg)
    total = len(jobs)
    LOG.info(f"{total} segment(s) queued for Gemma 4 transcription + translation + emotion.")
    LOG.info(f"Per-segment interaction log: {log_path}")

    if total == 0:
        save_json(cfg.conversation_json, [])
        save_json(cfg.translated_json, [])
        return []

    conversation = []
    success = failed = 0
    started = time.time()

    for idx, job in enumerate(jobs, start=1):
        segment_id = job["segment_id"]
        LOG.info(f"[{idx}/{total}] {job['speaker_id']} | {segment_id}")

        try:
            result = engine.transcribe_translate_segment(job, cfg.gemma_max_audio_seconds)
        except Exception as e:
            LOG.error(f"  {segment_id}: request failed - {e}")
            result = {
                "segment_id": segment_id, "is_song": False, "contains_dialogue": True,
                "dialogue_segments": [], "original_text": "",
                "detected_language": job.get("source_lang_code") or "",
                "translation": "", "emotion": "neutral",
                "ok": False, "error": str(e),
            }

        if not result.get("ok", False):
            LOG.warning(f"  {segment_id}: {result.get('error')}")
        else:
            tag = []
            if result.get("is_song"):
                tag.append("SONG")
            if result.get("contains_dialogue"):
                tag.append("dialogue")
            LOG.info(
                f"  -> [{'+'.join(tag) or 'none'}] [{result.get('detected_language')}] "
                f"\"{result.get('original_text', '')[:60]}\" | "
                f"emotion={result.get('emotion')} | "
                f"translation=\"{result.get('translation', '')[:60]}\""
            )

        original_volume_dbfs = _measure_original_volume_dbfs(job["audio_file"])
        dialogue_windows = _dialogue_windows_seconds(result.get("dialogue_segments", []), job["duration"])

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
            "emotion": result.get("emotion", "neutral"),
            "gender": job.get("gender"),
            "age_group": None,
            "confidence": None,
            "words": [],  # word-level timestamps are a faster-whisper-only feature; unused downstream
            "asr_translation_engine": f"ollama:{cfg.gemma_model_id}",
            "translation_ok": bool(result.get("ok", False)),
            "merged_from_count": job.get("merged_from_count", 1),
            "original_segments": job.get("original_segments"),
            # Song vs. dialogue (step14 acts on these - see cfg.keep_song_audio_original)
            "is_song": bool(result.get("is_song", False)),
            "contains_dialogue": bool(result.get("contains_dialogue", True)),
            # Absolute seconds *within this segment's own clip* (0 = the clip's own start,
            # not the video's timeline) where spoken dialogue falls, for a mixed song+dialogue
            # clip - Gemma's best-effort estimate, only meaningful when both flags above are true.
            "dialogue_windows": dialogue_windows,
            # Original per-segment loudness (step14 acts on this - see cfg.volume_aware_normalization)
            "original_volume_dbfs": original_volume_dbfs,
        }
        conversation.append(entry)

        if result.get("ok", False):
            success += 1
        else:
            failed += 1

        save_json(cfg.conversation_json, conversation)
        save_json(cfg.translated_json, conversation)

    if cfg.ollama_unload_after_step:
        LOG.info("Asking Ollama to unload the model (freeing RAM/VRAM before step14)...")
        engine.unload()

    LOG.info(
        f"Transcription + translation + emotion done in {time.time() - started:.2f}s | "
        f"success={success} failed={failed}"
    )

    del engine
    free_memory()

    return conversation


if __name__ == "__main__":
    run(PipelineConfig())
