#!/usr/bin/env python3
"""
End-to-end video dubbing / translation pipeline.

Usage
-----
    python run_pipeline.py --input input/video.mp4 --target-lang hi

    # Resume from a specific step (1-17) after a crash / to re-run just part
    # of the pipeline:
    python run_pipeline.py --input input/video.mp4 --target-lang hi --from-step 12 --to-step 17

See README.md for the full list of flags and one-time setup (one venv for
the pipeline, plus Ollama - a separate local server, not a venv - serving
Gemma 4 for step12 - and a HuggingFace token for pyannote).
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Must happen BEFORE numpy/torch/scipy/librosa are imported by anything else
# below (including src.utils, config, or any step module) - these libraries
# size their BLAS/OpenMP thread pools once, the first time their native
# backend is touched, so setting the env var after that point is too late to
# have any effect. This alone is often the single biggest lever for CPU
# speed on multi-core machines, since these libraries otherwise sometimes
# default to conservative or badly-oversubscribed thread counts.
# ---------------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(os.cpu_count() or 4))

import argparse
import sys
import time
from pathlib import Path

from config import PipelineConfig, XTTS_SUPPORTED_LANGUAGES
from src.utils import configure_cpu_threads, free_memory, get_logger

LOG = get_logger("pipeline")

STEPS = [
    (1, "Extract audio (mono + hq stereo)", "src.step01_extract_audio"),
    (2, "Audio metadata", "src.step02_audio_metadata"),
    (3, "Audio enhancement (noisereduce)", "src.step03_audio_enhancement"),
    (4, "Background music/SFX separation (Demucs)", "src.step04_background_extraction"),
    (5, "Voice activity detection (Silero VAD)", "src.step05_vad"),
    (6, "Speaker diarization (pyannote)", "src.step06_diarization"),
    (7, "Merge tiny same-speaker segments", "src.step07_segment_merge"),
    (8, "Split audio by speaker/segment", "src.step08_split_audio"),
    (9, "Speaker aggregation", "src.step09_speaker_aggregation"),
    (10, "Audio normalization", "src.step10_audio_normalization"),
    (11, "Speaker embedding / gender / age / emotion", "src.step11_speaker_embedding"),
    (12, "Transcription + translation + emotion (Gemma 4, via Ollama)", "src.step12_transcription_translation"),
    (13, "Reference voice preparation (multi-clip)", "src.step13_reference_voice"),
    (14, "Voice generation (chunked cloned dubbing)", "src.step14_voice_generation"),
    (15, "Subtitle generation (original/translated/bilingual)", "src.step15_subtitles"),
    (16, "Audio assembly (time-aligned dub + BGM/SFX)", "src.step16_audio_assembly"),
    (17, "Final video mux", "src.step17_video_mux"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Translate & dub a video with cloned voices.")

    p.add_argument("--input", type=Path, default=Path("input/video.mp4"), help="Path to source video")
    p.add_argument("--work-dir", type=Path, default=Path("output"), help="Root output directory")

    p.add_argument("--source-lang", default="auto", help="ISO 639-1 source language, or 'auto' to detect")
    p.add_argument("--target-lang", default="hi", help="ISO 639-1 target language (e.g. hi, es, fr, ta, ...)")

    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--cpu-threads", type=int, default=0)

    p.add_argument("--gemma-model-id", default="gemma4:e2b",
                    help="Ollama model tag for the step12 transcription/translation/emotion model "
                         "(default: gemma4:e2b; gemma4:e4b for higher quality) - "
                         "pull it first with `ollama pull <tag>`, see README section 1b")
    p.add_argument("--ollama-host", default=None,
                    help="Ollama server URL (default: $OLLAMA_HOST or http://localhost:11434)")

    p.add_argument("--hf-token", default=None,
                    help="HuggingFace token (or set HF_TOKEN env var) - needed for pyannote "
                         "(diarization/embedding)")

    p.add_argument("--subtitle-mode", choices=["translated", "original", "bilingual", "all"],
                    default="translated")
    p.add_argument("--no-original-audio-track", action="store_true",
                    help="Don't keep the original audio as a second track in the output")
    p.add_argument("--no-soft-subtitles", action="store_true", help="Don't embed subtitle tracks")
    p.add_argument("--burn-subtitles", action="store_true",
                    help="Also render an extra MP4 with subtitles burned into the picture")

    p.add_argument("--from-step", type=int, default=1)
    p.add_argument("--to-step", type=int, default=17)

    return p.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig(
        input_video=args.input,
        work_dir=args.work_dir,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        device=args.device,
        cpu_threads=args.cpu_threads,
        gemma_model_id=args.gemma_model_id,
        subtitle_mode=args.subtitle_mode,
        keep_original_audio_track=not args.no_original_audio_track,
        embed_soft_subtitles=not args.no_soft_subtitles,
        burn_in_subtitles=args.burn_subtitles,
    )
    if args.ollama_host:
        cfg.ollama_host = args.ollama_host
    if args.hf_token:
        cfg.hf_token = args.hf_token
    return cfg


def check_target_language(cfg: PipelineConfig) -> None:
    if cfg.target_lang not in XTTS_SUPPORTED_LANGUAGES:
        LOG.warning(
            f"'{cfg.target_lang}' is not one of XTTS-v2's 17 cloning languages "
            f"({sorted(XTTS_SUPPORTED_LANGUAGES)}). Step 13 will automatically fall back "
            f"to MMS-TTS with a gender-shifted single voice for this language."
        )


def main() -> int:
    args = parse_args()
    cfg = build_config(args)
    cfg.make_dirs()

    check_target_language(cfg)

    threads = configure_cpu_threads(cfg.cpu_threads)
    LOG.info(f"Input video : {cfg.input_video}")
    LOG.info(f"Target lang : {cfg.target_lang}")
    LOG.info(f"Work dir    : {cfg.work_dir}")
    LOG.info(f"Steps       : {args.from_step} -> {args.to_step}")
    LOG.info(f"CPU threads : {threads} (torch); BLAS envs default to {os.cpu_count() or 4}")

    overall_start = time.time()

    for step_num, title, module_name in STEPS:
        if step_num < args.from_step or step_num > args.to_step:
            continue

        import importlib
        module = importlib.import_module(module_name)

        LOG.info("")
        LOG.info(f"### STEP {step_num:02d}/{len(STEPS):02d} - {title} ###")
        step_start = time.time()

        try:
            module.run(cfg)
        except Exception as e:
            LOG.error(f"Step {step_num} ({title}) failed: {e}")
            LOG.error(
                f"Fix the issue and resume with: "
                f"python run_pipeline.py --input {cfg.input_video} --target-lang {cfg.target_lang} "
                f"--from-step {step_num} --to-step {args.to_step}"
            )
            free_memory()
            return 1

        LOG.info(f"Step {step_num} finished in {time.time() - step_start:.2f}s")

        # Release this step's model(s)/buffers (RAM and, if used, VRAM)
        # before the next step starts, so it gets a clean baseline instead
        # of competing with whatever the previous step left cached - see
        # free_memory()'s docstring in src/utils.py for exactly what this
        # does and why gc.collect() alone isn't enough.
        del module
        free_memory()

    LOG.info("")
    LOG.info("=" * 70)
    LOG.info(f"PIPELINE COMPLETE in {time.time() - overall_start:.2f}s")
    if cfg.final_video.exists():
        LOG.info(f"Final video: {cfg.final_video}")
    LOG.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
