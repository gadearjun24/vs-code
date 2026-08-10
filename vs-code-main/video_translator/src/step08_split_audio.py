"""
Step 08 - Split the clean-vocals audio into per-speaker segment clips.

Reads `merged_diarization.json` (step07's output) rather than raw
diarization directly, so each extracted clip already corresponds to a
natural-length, same-speaker block instead of a possibly-tiny raw
fragment. Extraction uses each block's `clip_start`/`clip_end` (which may
be a little wider than `start`/`end` for blocks step07 padded for Gemma's
benefit - see step07's docstring) - everything downstream that cares about
placement/sync/timing budgets still uses the block's true `start`/`end`
from merged_diarization.json, not the clip window.
"""

from __future__ import annotations

import soundfile as sf

from config import PipelineConfig
from src.utils import get_logger, load_json, resolve_speech_source

LOG = get_logger(__name__)


def run(cfg: PipelineConfig) -> None:
    source = resolve_speech_source(cfg)
    if not source.exists():
        raise FileNotFoundError(source)
    if not cfg.merged_diarization_json.exists():
        raise FileNotFoundError(
            f"{cfg.merged_diarization_json} not found - run step07 (segment merge) first."
        )

    audio, sr = sf.read(source)
    segments = load_json(cfg.merged_diarization_json)

    LOG.info(f"Loaded {len(segments)} merged segment block(s) (source: {source.name})")

    for seg in segments:
        folder = cfg.dir_speakers / seg["speaker_id"]
        folder.mkdir(parents=True, exist_ok=True)

        clip_start = seg.get("clip_start", seg["start"])
        clip_end = seg.get("clip_end", seg["end"])

        start_sample = max(0, int(clip_start * sr))
        end_sample = min(len(audio), int(clip_end * sr))
        clip = audio[start_sample:end_sample]

        sf.write(folder / f"{seg['segment_id']}.wav", clip, sr)

    LOG.info(f"Speaker folders saved in {cfg.dir_speakers}")


if __name__ == "__main__":
    run(PipelineConfig())
