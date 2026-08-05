#!/usr/bin/env python3
"""
End-to-end video dubbing / translation pipeline.

Usage
-----
    python run_pipeline.py --input input/video.mp4 --target-lang hi

    # Resume from a specific step (1-16) after a crash / to re-run just part
    # of the pipeline:
    python run_pipeline.py --input input/video.mp4 --target-lang hi --from-step 11 --to-step 16

See README.md for the full list of flags and one-time setup (two venvs -
one for the main pipeline, one for the Gemma 4 transcription/translation
step - plus a HuggingFace token for pyannote + Gemma 4).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from config import PipelineConfig, XTTS_SUPPORTED_LANGUAGES
from src.utils import get_logger

LOG = get_logger("pipeline")

STEPS = [
    (1, "Extract audio (mono + hq stereo)", "src.step01_extract_audio"),
    (2, "Audio metadata", "src.step02_audio_metadata"),
    (3, "Audio enhancement (noisereduce)", "src.step03_audio_enhancement"),
    (4, "Background music/SFX separation (Demucs)", "src.step04_background_extraction"),
    (5, "Voice activity detection (Silero VAD)", "src.step05_vad"),
    (6, "Speaker diarization (pyannote)", "src.step06_diarization"),
    (7, "Split audio by speaker/segment", "src.step07_split_audio"),
    (8, "Speaker aggregation", "src.step08_speaker_aggregation"),
    (9, "Audio normalization", "src.step09_audio_normalization"),
    (10, "Speaker embedding / gender / age / emotion", "src.step10_speaker_embedding"),
    (11, "Transcription + translation (Gemma 4 E2B, audio-native)", "src.step11_transcription_translation"),
    (12, "Reference voice preparation (multi-clip)", "src.step12_reference_voice"),
    (13, "Voice generation (chunked cloned dubbing)", "src.step13_voice_generation"),
    (14, "Subtitle generation (original/translated/bilingual)", "src.step14_subtitles"),
    (15, "Audio assembly (time-aligned dub + BGM/SFX)", "src.step15_audio_assembly"),
    (16, "Final video mux", "src.step16_video_mux"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Translate & dub a video with cloned voices.")

    p.add_argument("--input", type=Path, default=Path("input/video.mp4"), help="Path to source video")
    p.add_argument("--work-dir", type=Path, default=Path("output"), help="Root output directory")

    p.add_argument("--source-lang", default="auto", help="ISO 639-1 source language, or 'auto' to detect")
    p.add_argument("--target-lang", default="hi", help="ISO 639-1 target language (e.g. hi, es, fr, ta, ...)")

    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--cpu-threads", type=int, default=0)

    p.add_argument("--gemma-model-id", default="google/gemma-4-E2B-it",
                    help="HuggingFace model id for the step11 transcription/translation model "
                         "(default: google/gemma-4-E2B-it; google/gemma-4-E4B-it for higher quality)")
    p.add_argument("--gemma-device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="Device used inside the separate Gemma worker venv")

    p.add_argument("--hf-token", default=None,
                    help="HuggingFace token (or set HF_TOKEN env var) - needed for pyannote "
                         "(diarization/embedding) and for google/gemma-4-E2B-it (accept its "
                         "license on Hugging Face first)")

    p.add_argument("--subtitle-mode", choices=["translated", "original", "bilingual", "all"],
                    default="translated")
    p.add_argument("--no-original-audio-track", action="store_true",
                    help="Don't keep the original audio as a second track in the output")
    p.add_argument("--no-soft-subtitles", action="store_true", help="Don't embed subtitle tracks")
    p.add_argument("--burn-subtitles", action="store_true",
                    help="Also render an extra MP4 with subtitles burned into the picture")

    p.add_argument("--from-step", type=int, default=1)
    p.add_argument("--to-step", type=int, default=16)

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
        gemma_device=args.gemma_device,
        subtitle_mode=args.subtitle_mode,
        keep_original_audio_track=not args.no_original_audio_track,
        embed_soft_subtitles=not args.no_soft_subtitles,
        burn_in_subtitles=args.burn_subtitles,
    )
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

    LOG.info(f"Input video : {cfg.input_video}")
    LOG.info(f"Target lang : {cfg.target_lang}")
    LOG.info(f"Work dir    : {cfg.work_dir}")
    LOG.info(f"Steps       : {args.from_step} -> {args.to_step}")

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
            return 1

        LOG.info(f"Step {step_num} finished in {time.time() - step_start:.2f}s")

    LOG.info("")
    LOG.info("=" * 70)
    LOG.info(f"PIPELINE COMPLETE in {time.time() - overall_start:.2f}s")
    if cfg.final_video.exists():
        LOG.info(f"Final video: {cfg.final_video}")
    LOG.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
