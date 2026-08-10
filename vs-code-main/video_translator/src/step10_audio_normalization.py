"""Step 10 - Loudness-normalize each per-speaker audio file (EBU R128)."""

from __future__ import annotations

import time
from pathlib import Path

from config import PipelineConfig
from src.utils import get_logger, load_json, run_ffmpeg, save_json

LOG = get_logger(__name__)


def normalize_audio(input_file: Path, output_file: Path, sample_rate: int):
    run_ffmpeg([
        "-i", str(input_file),
        "-ar", str(sample_rate),
        "-ac", "1",
        "-sample_fmt", "s16",
        "-af", "loudnorm=I=-16:LRA=11:TP=-1.5",
        str(output_file),
    ])


def run(cfg: PipelineConfig) -> list:
    if not cfg.speaker_profiles_json.exists():
        raise FileNotFoundError(cfg.speaker_profiles_json)

    cfg.dir_normalized_speakers.mkdir(parents=True, exist_ok=True)

    speakers = load_json(cfg.speaker_profiles_json)
    normalized = []
    start = time.time()

    for index, speaker in enumerate(speakers, start=1):
        input_audio = Path(speaker["audio_file"])
        output_audio = cfg.dir_normalized_speakers / input_audio.name

        LOG.info(f"[{index}/{len(speakers)}] {speaker['speaker_id']}")
        normalize_audio(input_audio, output_audio, cfg.sample_rate)

        normalized.append({
            "speaker_id": speaker["speaker_id"],
            "audio_file": str(output_audio),
            "sample_rate": cfg.sample_rate,
            "channels": 1,
            "sample_format": "PCM16",
            "segments": speaker["segments"],
            "duration": speaker["duration"],
            "normalized": True,
        })

    save_json(cfg.normalized_speakers_json, normalized)
    LOG.info(f"Normalized {len(normalized)} speakers in {time.time() - start:.2f}s")
    return normalized


if __name__ == "__main__":
    run(PipelineConfig())
